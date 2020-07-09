#!/usr/bin/env python3

import os
import sys
import time
import threading
from multiprocessing import Process

import tensorflow as tf
import numpy

sys.path.append("../bin/")

import kuiba_op_v2
# kuiba_pybind 用于获取不同训练目标(loss_function_name)的parameter num，input dim，output dim
import kuiba_pybind
import kuiba_utils
import worker_cpu_affinity


tf.flags.DEFINE_string("flag_file", "server_static.flags", "flag file")
tf.flags.DEFINE_string("kuiba_op_library_file", "libkuiba_op.so", "kuiba op library path")
tf.flags.DEFINE_string("mode", "train", "running mode")
tf.flags.DEFINE_string("predict_graph_dir", "../predict_graph", "predict graph dir")
tf.flags.DEFINE_string("predict_graph_version", "", "predict graph version")
tf.flags.DEFINE_integer("gpu_index", 0, "gpu index")
tf.flags.DEFINE_integer("worker_index", 0, "gpu index")
FLAGS = tf.flags.FLAGS

numpy.set_printoptions(edgeitems=10)


global_step = 0
global_step_lock = threading.Lock()

# 多种loss支持不同业务场景
lhuc_prefix = "lhuc_"
ftr_loss_name = "ftr_n"
ctr_loss_name = "ctr_n"
ltr_loss_name = "ltr_n"
lvtr_loss_name = "lvtr_n"
wtr_loss_name = "wtr_n"
lhuc_ctr_loss_name = lhuc_prefix + ctr_loss_name
# 进程数
worker_num = 3
# 进程中线程数量
core_num_per_worker = 7
# NOTE 应该是进行最佳cpu分配
worker_cpu_affinity_result = worker_cpu_affinity.get_worker_cpu_affinity(worker_num, core_num_per_worker)


def fm_cross_layer(input, dropout_keep, batch_norm, train_phase):
    # _________ sum_square part _____________

    summed_features_emb = tf.reduce_sum(input, 1)  # None * K
    # get the element-multiplication
    summed_features_emb_square = tf.square(summed_features_emb)  # None * K

    # _________ square_sum part _____________
    squared_features_emb = tf.square(input)
    squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)  # None * K

    # ________ FM __________
    output = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  # None * K
    if batch_norm:
        output = batch_norm_layer(output, train_phase=train_phase)
    if dropout_keep:
        output = tf.nn.dropout(output, dropout_keep)  # dropout at the FM layer

    return output


def fm_model_build(input, weight):



    pass





class FM(object):

    def __init__(self, features_M, features_C, pretrain_flag, save_file, hidden_factor, loss_type, epoch, batch_size,
                 learning_rate, lambda_bilinear, keep, optimizer_type, batch_norm, verbose, log_dir, hist_nums, pretrain_embedding,
                 params, rank=True, random_seed=2019):
        pass

    def fm_cross_layer(self, parameters):
        # _________ square_sum part _____________
        summed_features_emb = tf.reduce_sum(parameters, 1)
        # get the element-multiplication
        summed_features_emb_square = tf.square(summed_features_emb)  # None * K

        # _________ square_sum part _____________
        squared_features_emb = tf.square(input)
        squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)  # None * K

        output = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  # None * K
        return output

    def fm_layer(self, global_bias, list_bias, embedding_parameters):
        corss_output = self.fm_cross_layer(embedding_parameters)
        bias = global_bias * tf.ones()










    def build(self):
        pass


    def fm_cross_layer(self, input, dropout_keep, weight, batch_norm, train_phase):

        # _________ square_sum part _____________
        summed_features_emb = tf.reduce_sum(input, 1)  # None * K
        # get the element-multiplication
        summed_features_emb_square = tf.square(summed_features_emb)  # None * K

        # _________ square_sum part _____________
        squared_features_emb = tf.square(input)
        squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)  # None * K

        # ________ FM __________
        output = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  # None * K
        if batch_norm:
            output = batch_norm_layer(output, train_phase=train_phase)
        if dropout_keep:
            output = tf.nn.dropout(output, dropout_keep)  # dropout at the FM layer
        return output


    def fm_linear_layer(self, input):
        self.Feature_bias = tf.reduce_sum(
            tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_features), 1)









def batch_norm_layer(x, train_phase):
    bn_train = tf.layers.batch_normalization(x, momentum=0.9, center=True, scale=True,
                                             trainable=True, reuse=None, training=True)
    bn_inference = tf.layers.batch_normalization(x, momentum=0.9, center=True, scale=True,
                                                 trainable=True, reuse=None, training=False)
    z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
    return z





def lhuc_layer(output, size):
    output = tf.layers.dense(output, 256, activation=tf.nn.relu)
    output = tf.layers.dense(output, size, activation=tf.nn.sigmoid)
    # 标量 * 张量
    output = tf.scalar_mul(tf.constant(2.0), output)
    return output


def hidden_layers(name, hidden_size_list, output):
    layer_index = 1
    for hidden_size in hidden_size_list:
        layer_prefix = name + "_layer"
        output = tf.layers.dense(output, hidden_size, activation=tf.nn.relu)
        if kuiba_utils.train_mode():
            tf.summary.histogram("{}_{}_output".format(layer_prefix, layer_index), output)
        layer_index += 1
    return output


def lhuc_hidden_layers(name, hidden_size_list, dnn_input):
    keys = list(dnn_input.keys())
    keys.sort()
    main_input_keys = [key for key in keys if not key.startswith(lhuc_prefix)]

    lhuc_input = tf.concat([dnn_input[key] for key in keys], axis=1)
    main_output = tf.concat([dnn_input[key] for key in main_input_keys], axis=1)
    if kuiba_utils.train_mode():
        tf.summary.histogram("{}_input_before_lhuc".format(name), main_output)

    # lhuc to input
    main_input_size = sum([kuiba_pybind.get_parameter_output_dim(key) for key in main_input_keys])
    with tf.variable_scope(name + "lhuc_input", reuse=kuiba_utils.reuse_variables()):
        lhuc_output = lhuc_layer(lhuc_input, main_input_size)
    main_output = tf.multiply(main_output, lhuc_output)

    # lhuc to hidden layers
    layer_index = 1
    for hidden_size in hidden_size_list:
        layer_prefix = "{}_{}_hidden".format(name, layer_index)
        main_output = tf.layers.dense(main_output, hidden_size, activation=tf.nn.relu)
        if kuiba_utils.train_mode():
            tf.summary.histogram("{}_before_lhuc".format(layer_prefix), main_output)
        with tf.variable_scope(layer_prefix + "_lhuc", reuse=kuiba_utils.reuse_variables()):
            lhuc_output = lhuc_layer(lhuc_input, hidden_size)
        if kuiba_utils.train_mode():
            tf.summary.histogram("{}_lhuc".format(layer_prefix), lhuc_output)
        main_output = tf.multiply(main_output, lhuc_output)
        if kuiba_utils.train_mode():
            tf.summary.histogram("{}_after_lhuc".format(layer_prefix), main_output)
        layer_index += 1
    return main_output


def lwtr_shared_layers(dnn_input):
    with tf.variable_scope("lwtr_shared_layers", reuse=kuiba_utils.reuse_variables()):
        output = hidden_layers("lwtr", [512, 256, 128, 128, 128], dnn_input)
    return output


def vftr_shared_layers(dnn_input):
    with tf.variable_scope("vftr_shared_layers", reuse=kuiba_utils.reuse_variables()):
        output = hidden_layers("vftr", [512, 256, 256, 128, 128], dnn_input)
    return output


def trans(dnn_input):
    direct_params = list(dnn_input.keys())
    direct_params.sort()
    return tf.concat([dnn_input[param] for param in direct_params], axis=1)


def lhuc_ctr_model(dnn_input):
    with tf.variable_scope("lhuc_ctr_layers", reuse=kuiba_utils.reuse_variables()):
        hidden_output = lhuc_hidden_layers(lhuc_ctr_loss_name, [512, 512, 256, 128, 128], dnn_input)
        output = tf.layers.dense(hidden_output, 1, activation=tf.nn.sigmoid)
        if kuiba_utils.train_mode():
            tf.summary.histogram("lhuc_{}_predict".format(lhuc_ctr_loss_name), output)
    if kuiba_utils.predict_mode():
        output = tf.identity(output, lhuc_ctr_loss_name)
    return output


def ctr_model(dnn_input):
    dnn_input = {key: value for key, value in dnn_input.items() if not key.startswith(lhuc_prefix)}
    # 特征拼接作为输入
    dnn_input = trans(dnn_input)
    with tf.variable_scope("ctr_layers", reuse=kuiba_utils.reuse_variables()):
        dnn_input = hidden_layers(
            ctr_loss_name, [512, 512, 256, 128, 128], dnn_input)
        output = tf.layers.dense(dnn_input, 1, activation=tf.nn.sigmoid)
        if kuiba_utils.train_mode():
            tf.summary.histogram("{}_predict".format(ctr_loss_name), output)
    if kuiba_utils.predict_mode():
        output = tf.identity(output, "ctr_n")
    return output


def ltr_model(dnn_input):
    dnn_input = {key: value for key, value in dnn_input.items() if not key.startswith(lhuc_prefix)}
    dnn_input = trans(dnn_input)
    shared_output = lwtr_shared_layers(dnn_input)
    with tf.variable_scope("ltr_layers", reuse=kuiba_utils.reuse_variables()):
        output = tf.layers.dense(shared_output, 1, activation=tf.nn.sigmoid)
        if kuiba_utils.train_mode():
            tf.summary.histogram("{}_predict".format(ltr_loss_name), output)
    if kuiba_utils.predict_mode():
        output = tf.identity(output, "ltr_n")
    return output


def wtr_model(dnn_input):
    dnn_input = {key: value for key, value in dnn_input.items() if not key.startswith(lhuc_prefix)}
    dnn_input = trans(dnn_input)
    shared_output = lwtr_shared_layers(dnn_input)
    with tf.variable_scope("wtr_layers", reuse=kuiba_utils.reuse_variables()):
        output = tf.layers.dense(shared_output, 1, activation=tf.nn.sigmoid)
        if kuiba_utils.train_mode():
            tf.summary.histogram("{}_predict".format(wtr_loss_name), output)
    if kuiba_utils.predict_mode():
        output = tf.identity(output, "wtr_n")
    return output


def lvtr_model(dnn_input):
    dnn_input = {key: value for key, value in dnn_input.items() if not key.startswith(lhuc_prefix)}
    dnn_input = trans(dnn_input)
    shared_output = vftr_shared_layers(dnn_input)
    with tf.variable_scope("lvtr_layers", reuse=kuiba_utils.reuse_variables()):
        output = tf.layers.dense(shared_output, 1, activation=tf.nn.sigmoid)
        if kuiba_utils.train_mode():
            tf.summary.histogram("{}_predict".format(lvtr_loss_name), output)
    if kuiba_utils.predict_mode():
        output = tf.identity(output, "lvtr_n")
    return output


def ftr_model(dnn_input):
    dnn_input = {key: value for key, value in dnn_input.items() if not key.startswith(lhuc_prefix)}
    dnn_input = trans(dnn_input)
    shared_output = vftr_shared_layers(dnn_input)
    with tf.variable_scope("ftr_layers", reuse=kuiba_utils.reuse_variables()):
        output = tf.layers.dense(shared_output, 1, activation=tf.nn.sigmoid)
        if kuiba_utils.train_mode():
            tf.summary.histogram("{}_predict".format(ftr_loss_name), output)
    if kuiba_utils.predict_mode():
        output = tf.identity(output, "ftr_n")
    return output


def gen_dryrun_input(loss_name):
    # 见配置文件
    parameters = kuiba_pybind.get_parameter_names(loss_name)
    dims = [kuiba_pybind.get_parameter_output_dim(param) for param in parameters]
    return {param: tf.placeholder(tf.float32, shape=[None, dim])
            for param, dim in zip(parameters, dims)}


# 不同训练目标对应训练模型字典
loss_model_dict = {
    ctr_loss_name: ctr_model,
    ltr_loss_name: ltr_model,
    # wtr_loss_name: wtr_model,
    # ftr_loss_name: ftr_model,
    lvtr_loss_name: lvtr_model,
    # lhuc_ctr_loss_name: lhuc_ctr_model
}


def dryrun():
    '''
        测试
    '''
    kuiba_utils.set_dryrun_to_get_variables(True)
    for loss_name, model in loss_model_dict.items():
        dryrun_input = gen_dryrun_input(loss_name)
        _ = model(dryrun_input)
    kuiba_utils.set_dryrun_to_get_variables(False)


def get_bp_parameters(loss_name, parameter_dict):
    '''

    :param loss_name: 损失函数name
    :param parameter_dict: loss_name: (参数列表， label)
    :return:
    '''
    # 取出所有参数name list
    parameter_name_list = list(parameter_dict.keys())
    # 取出所有参数val list
    parameter_tensor_list = list(parameter_dict.values())
    # 以lhuc_prefix打头需要特殊处理
    if loss_name.startswith(lhuc_prefix):
        # 取出指定loss对应的参数
        bp_dict = {key: value for key, value in parameter_dict.items() if key.startswith(lhuc_prefix)}
        parameter_name_list = list(bp_dict.keys())
        parameter_tensor_list = list(bp_dict.values())
    print("bp_params:%s => %s" % (loss_name, parameter_name_list))
    return parameter_name_list, parameter_tensor_list


# 任务入口函数
def main(_):
    cmd_mode = FLAGS.mode
    FLAGS.mode = kuiba_utils.get_predict_mode()
    # 单进程0，测试work
    predict_process = Process(target=work_process, args=(0,))
    predict_process.start()
    predict_process.join()

    # FLAGS.mode = cmd_mode
    # 指定训练模式为train
    FLAGS.mode = "train"
    # work_index: process
    worker_map = {}
    # 生成多进程列表 nums=3
    for worker_index, cpus in worker_cpu_affinity_result.items():
        worker = Process(target=work_process, args=(worker_index,))
        worker_map[worker_index] = worker
        worker.start()
        pid = worker.pid
        # 进程和cpu绑定
        os.system('taskset -pc ' + cpus + ' ' + str(pid))
    # 分布式训练中，train不会终止，而是到达一定时间后进行eval等操作, TODO 数据如何过期？
    while len(worker_map) > 0:
        for worker_index in worker_map.keys():
            worker = worker_map[worker_index]
            # 训练进程异常则重启，并从最近一次保存的节点恢复
            if not worker.is_alive():
                print("process is not alive, restart, worker_index:", worker_index)
                worker = Process(target=work_process, args=(worker_index,))
                worker_map[worker_index] = worker
                worker.start()
        time.sleep(0.5)
    print("main process finish!")


def work_process(worker_index):
    # 生成当前进程的name
    worker_name = "tf_worker_" + str(worker_index)
    # TODO api q：作用？
    kuiba_pybind.init(worker_name, FLAGS.flag_file, "")
    dryrun()
    # 网络参数列表，见dynamic_json_config.json
    network_var_list = kuiba_utils.get_network_variables()
    # 网络参数名列表
    network_var_names = [var.name for var in network_var_list]
    # summary加入参数信息并打印
    for i in range(len(network_var_names)):
        # 直方图显示，反映变量的分布情况
        tf.summary.histogram("{}".format(network_var_names[i]), network_var_list[i])
        print(network_var_list[i])
    # 刷新输出缓存=》输出
    sys.stdout.flush()

    x = tf.placeholder(tf.int64, shape=(), name="x")
    # 初始化一个minibatch的训练：包括读取训练样本，获取网络参数和parameter值等，返回值：batch_id标志这一组minibatch样本
    batch_id_tensor = kuiba_op_v2.start_batch_op(x)
    # loss name: (参数字典， label)
    loss_data = {}
    for loss_name in loss_model_dict.keys():
        # 拉取对应模型（loss_name）的embedding input，样本label -> 特征向量化 embedding_lookup TODO api q：ignore返回信息待确认
        parameter_dict, label, *ignore = kuiba_op_v2.pull_sparse_op(batch_id_tensor, loss_name)
        loss_data[loss_name] = (parameter_dict, label)
    # 从ps pull指定minibatch的网络参数 -> ps端网络参数同步
    pull_network_tensor = kuiba_op_v2.pull_network_op(batch_id_tensor, network_var_list)

    loss_tensor_dict = {}
    batch_finish_dep = []
    with tf.control_dependencies([pull_network_tensor]):
        for loss_name, model in loss_model_dict.items():
            parameter_dict, labels = loss_data[loss_name]
            bp_param_name_list, bp_param_tensor_list = get_bp_parameters(loss_name, parameter_dict)
            # 模型输出
            xtr_output = model(parameter_dict)
            # 训练模式
            if kuiba_utils.train_mode():
                # 构建loss， 即所有的model的loss是相同的...
                xtr_loss = tf.losses.log_loss(labels=labels, predictions=xtr_output, reduction=tf.losses.Reduction.SUM)
                # loss 存储
                loss_tensor_dict[loss_name] = xtr_loss
                # 计算梯度
                # 返回值： embedding梯度， 网络参数， 网络参数梯度
                parameter_grad, network_vars, network_grad = kuiba_utils.compute_grad(
                    xtr_loss, bp_param_tensor_list, network_var_list)
                # 更新指定minibatch的属于loss_function_name的所有parameter的梯度值到ps
                push_parameter_tensor = kuiba_op_v2.push_partial_sparse_op(
                    batch_id_tensor, parameter_grad, loss_name, bp_param_name_list)
                # 更新指定minibatch的属于loss_function_name的网络参数梯度到ps
                push_network_tensor = kuiba_op_v2.push_network_op(
                    batch_id_tensor, network_grad, loss_name, [var.name for var in network_vars])
                push_auc_tensor = kuiba_op_v2.push_auc_op(batch_id_tensor, xtr_output, xtr_loss, loss_name)
                batch_finish_dep.extend([push_parameter_tensor, push_network_tensor, push_auc_tensor])
            # 评估模式不用对ps端参数进行更新
            if kuiba_utils.evaluate_mode():
                xtr_loss = tf.losses.log_loss(labels=labels, predictions=xtr_output, reduction=tf.losses.Reduction.SUM)
                loss_tensor_dict[loss_name] = xtr_loss
                push_auc_tensor = kuiba_op_v2.push_auc_op(batch_id_tensor, xtr_output, xtr_loss, loss_name)
                batch_finish_dep.extend([push_auc_tensor])

    with tf.control_dependencies(batch_finish_dep):
        # 结束指定minibatch训练，释放资源
        finish_batch_tensor = kuiba_op_v2.finish_batch_op(batch_id_tensor)
        merge_summary = tf.summary.merge_all()

    # allow_soft_placement:软分配，cpu和gpu op分离，intra_op_parallelism_threads:线程池中线程数量，同一个op计算并行加速
    # inter_op_parallelism_threads：不同op并行
    config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=1, inter_op_parallelism_threads=10)
    # 可动态增加显存分配
    config.gpu_options.allow_growth = True
    # 每个进程可分配显存比例
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    loss_name_list = list(loss_model_dict.keys())
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # 预测模式
        if kuiba_utils.predict_mode():
            print("worker_index:", worker_index)
            if worker_index == 0:
                # 图结构保存到文件，两种格式的pb文件
                tf.train.write_graph(sess.graph_def, FLAGS.predict_graph_dir,
                                     'predict_graph.binary.pb.{}'.format(FLAGS.predict_graph_version), as_text=False)
                tf.train.write_graph(sess.graph_def, FLAGS.predict_graph_dir,
                                     'predict_graph.text.pb.{}'.format(FLAGS.predict_graph_version), as_text=True)
            # TODO api q：作用？
            kuiba_utils.verify_network_variables(network_var_list)
            sys.exit()

        # 从上一次保存点加载进行初始化
        with open('{}/predict_graph.text.pb.{}'.format(FLAGS.predict_graph_dir, FLAGS.predict_graph_version), 'r') as f:
            predict_graph_text = f.read()
            kuiba_utils.verify_predict_graph(predict_graph_text, *loss_name_list)
        with open('{}/predict_graph.binary.pb.{}'.format(FLAGS.predict_graph_dir, FLAGS.predict_graph_version),
                  'rb') as f:
            # 二进制保存图
            predict_graph_binary = f.read()

        # TODO api q: 创建TensorProto，参数：参数值，貌似是用来存储的
        network_init_weights = [tf.make_tensor_proto(var.eval(), dtype=tf.float32) for var in network_var_list]
        init_worker_tensor = kuiba_op_v2.init_worker_op(
            worker_name, FLAGS.flag_file, network_var_names, network_init_weights, predict_graph_binary)
        # 参数初始化
        sess.run(init_worker_tensor)

        kuiba_utils.verify_network_variables(network_var_list)

        print("------------original network------------")
        for var in network_var_list:
            print(var.name + ": " + str(var.shape))
            print(str(var.eval()))
            print("")

        if kuiba_utils.train_mode():
            summary_writer = tf.summary.FileWriter('../summary', sess.graph)
            # summary_writer = None
        if kuiba_utils.evaluate_mode():
            summary_writer = None

        threads = []
        # TODO 10个线程进行训练，与core_num_per_worker关系
        for i in range(10):
            thread = threading.Thread(target=train_thread, name="train-thread{}".format(i), args=(
                sess, loss_tensor_dict, finish_batch_tensor, x, merge_summary, summary_writer), daemon=True)
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()


def train_thread(sess, loss_tensor_dict, finish_batch_tensor, x, merge_summary, summary_writer):
    '''

    :param sess:
    :param loss_tensor_dict:
    :param finish_batch_tensor:
    :param x:
    :param merge_summary:
    :param summary_writer:
    :return:
    '''
    global global_step_lock
    global global_step
    cur_step = 0
    while True:
        with global_step_lock:
            global_step += 1
            cur_step = global_step
        start_time = time.time()
        loss_name_list = list(loss_tensor_dict.keys())
        loss_tensor_list = list(loss_tensor_dict.values())
        # run loss and train op
        ret_list = sess.run(loss_tensor_list + [finish_batch_tensor], feed_dict={x: 1})
        loss_value_list = ret_list[0: len(loss_tensor_list)]
        duration = time.time() - start_time
        loss_info = ",".join(
            "{}_loss: {}".format(name, value) for
            name, value in zip(loss_name_list, loss_value_list))
        print("step: {}, ctr_loss: {}, ".format(cur_step, loss_info))


if __name__ == "__main__":
    tf.app.run()
