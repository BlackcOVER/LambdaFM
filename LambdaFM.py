'''
Tensorflow implementation of lambdaRank Factorization Machines (LambdaFM) as described in:

@author:
Fan Li
@references:
https://github.com/tensorflow/ranking
https://github.com/hexiangnan/neural_factorization_machine
'''

import os
import time
import argparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing
import pandas as pd

from loss import create_ndcg_lambda_weight, PairwiseLogisticLoss
from load_data import *
from metrics import *


'''
Q1: Should'nt be different of the gradient direction of optimizer for rankNet and lambdaRank?
'''

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description="Run FM.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='test',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=10,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='flag for pretrain. 1: initialize from pretrain; 0: '
                             'randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default=0.1,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--keep_prob', type=float, default=0.5,
                        help='Keep probility (1-dropout_ratio) for the Bi-Interaction layer. 1: no dropout')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, '
                             'AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=1,
                        help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--log_dir', type=str, default='log',
                        help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--list_size', type=int, default=24,
                        help='max size of each group')
    parser.add_argument('--features_num', type=int, default=31,
                        help='max features num of each sample')
    parser.add_argument('--feature_id_num', type=int, default=182796,
                        help='features num of all sample')

    return parser.parse_args()


class LambdaFM(BaseEstimator, TransformerMixin):
    def __init__(self, feature_id_num, pretrain_flag, save_file, hidden_factor, epoch, batch_size,
                 learning_rate, lambda_bilinear, keep, optimizer_type, batch_norm, verbose, log_dir,
                 pretrain_embedding, list_size, features_num, rank=True, random_seed=2019):
        # bind params to class
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_factor = hidden_factor
        self.save_file = save_file
        self.pretrain_flag = pretrain_flag
        self.feature_id_num = feature_id_num
        self.lambda_bilinear = lambda_bilinear
        self.keep = keep
        self.epoch = epoch
        self.log_dir = log_dir
        self.list_size = list_size
        self.features_num = features_num
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.verbose = verbose
        # performance of each epoch
        self.train_ndcg, self.valid_ndcg, self.test_ndcg = list(), list(), list()
        self.pretrain_embedding = pretrain_embedding
        self.lambda_weight = create_ndcg_lambda_weight() if rank else None
        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:10'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.train_features = tf.placeholder(tf.int32, shape=[None, self.list_size, self.features_num])

            self.train_values = tf.placeholder(tf.float32, shape=[None, self.list_size, self.features_num])

            self.train_labels = tf.placeholder(tf.float32, shape=[None, self.list_size])

            self.train_features_flatten = tf.reshape(self.train_features,
                                                     shape=[-1, self.features_num])
            self.train_values_flatten = tf.reshape(self.train_values,
                                                   shape=[-1, self.features_num, 1])

            self.dropout_keep = tf.placeholder(tf.float32)
            self.train_phase = tf.placeholder(tf.bool)

            # Init variables.
            self.weights = self._initialize_weights()

            # Model.
            self.nonzero_embeddings = tf.nn.embedding_lookup(
                self.weights['feature_embeddings'], self.train_features_flatten) * self.train_values_flatten

            # _________ sum_square part _____________
            # get the summed up embeddings of features.
            self.summed_features_emb = tf.reduce_sum(self.nonzero_embeddings, 1)  # None * K
            # get the element-multiplication
            self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

            # _________ square_sum part _____________
            self.squared_features_emb = tf.square(self.nonzero_embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

            # ________ FM __________
            self.FM = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)  # None * K
            if self.batch_norm:
                self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase)
            self.FM = tf.nn.dropout(self.FM, self.dropout_keep)  # dropout at the FM layer None * K

            # _________out _________
            # 二阶求和
            Bilinear = tf.reduce_sum(self.FM, 1, keep_dims=True)  # None * 1
            # 一阶求和
            self.Feature_bias = tf.reduce_sum(
                tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_features_flatten) * self.train_values_flatten, 1)  # None * 1
            # 偏置
            Bias_tmp = tf.reshape(tf.ones_like(self.train_labels, dtype=tf.float32), shape=[-1, 1])
            Bias = self.weights['bias'] * Bias_tmp  # None * 1
            self.out = tf.add_n([Bilinear, self.Feature_bias, Bias])  # None * 1
            # logit
            self.logit = tf.reshape(self.out, shape=[-1, self.list_size])

            # Compute the lambdaRank NDCG loss.
            l2_regularizer = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            pairwise_loss_func = PairwiseLogisticLoss('fm_loss', self.lambda_weight)
            if self.lambda_bilinear > 0:
                self.loss = pairwise_loss_func.compute(self.train_labels, self.logit, None) + tf.add_n(l2_regularizer)
            else:
                self.loss = pairwise_loss_func.compute(self.train_labels, self.logit, None)

            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8)
            # tfranking 使用此优化器
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = \
                    tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.minimize(self.loss)
            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()  # shape is an array of tf.Dimension
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print(f'#params: {total_parameters}')

    def _initialize_weights(self):
        all_weights = dict()
        # 是否预加载词向量
        if self.pretrain_flag > 0:
            feature_embeddings = np.random.normal(0.0, 0.01, [self.feature_id_num, self.hidden_factor]).astype(np.float32)
            for index, vec in self.pretrain_embedding.items():
                feature_embeddings[index] = vec
            all_weights['feature_embeddings'] = \
                tf.get_variable(name='feature_embeddings', dtype=tf.float32, initializer=feature_embeddings,
                                regularizer=tf.keras.regularizers.l2(self.lambda_bilinear))
        else:
            all_weights['feature_embeddings'] = \
                tf.get_variable(name='feature_embeddings', shape=[self.feature_id_num, self.hidden_factor],
                                dtype=tf.float32, initializer=tf.random_normal_initializer(0.0, 0.01),
                                regularizer=tf.keras.regularizers.l2(self.lambda_bilinear))
        all_weights['feature_bias'] = \
            tf.get_variable(name='feature_bias', shape=[self.feature_id_num, 1],
                            initializer=tf.random_normal_initializer(0.0, 0.0))  # features_M * 1
        all_weights['bias'] = tf.get_variable(name='bias', initializer=tf.constant(0.0))  # 1 * 1
        return all_weights

    def batch_norm_layer(self, x, train_phase):
        # Note: the decay parameter is tunable
        bn_train = tf.layers.batch_normalization(x, momentum=0.9, center=True, scale=True,
                                                 trainable=True, reuse=None, training=True)
        bn_inference = tf.layers.batch_normalization(x, momentum=0.9, center=True, scale=True,
                                                     trainable=True, reuse=None, training=False)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.train_features: data['X'], self.train_labels: data['Y'],
                     self.train_values: data['X_value'], self.dropout_keep: self.keep, self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.train_op), feed_dict=feed_dict)
        print('train_loss=%.4f' % (loss))
        return loss

    def construct_dataset(self, X_, Y_, X_value):
        Data_Dic = dict()
        # 根据样本特征长度从大到小排列
        Data_Dic['Y'] = Y_
        Data_Dic['X'] = X_
        Data_Dic['X_value'] = X_value
        return Data_Dic

    def get_random_block_from_data(self, data, batch_size, step):  # generate a random block of training data
        # 每次随机生成batch
        # start_index = np.random.randint(0, len(data['Y']) - batch_size)
        start_index = step * batch_size
        X, Y, X_value = list(), list(), list()
        # forward get sample
        i = start_index
        count = 0
        # 针对X中每一条样本，有k条候选书籍
        while len(X) < batch_size and i < len(data['X']):
            Y.append(data['Y'][i])
            X.append(data['X'][i])
            X_value.append(data['X_value'][i])
            # 每一条样本中的每一本书籍
            i = i + 1
            count += 1
        return {'X': X, 'Y': Y, 'X_value': X_value}

    def train(self, data):  # fit a dataset
        # Check Init performance
        if self.verbose > 0:
            Train_data, Validation_data, Test_data = \
                data.construct_data(self.list_size, self.features_num, self.batch_size * 10)
            t2 = time.time()
            print('eval init start!')
            init_valid = self.evaluate(Validation_data)
            init_test = self.evaluate(Test_data)
            print("Init: \t validation=%.4f, test=%.4f [%.1f s]" % (
                 init_valid, init_test, time.time() - t2))
        # TODO 生成器
        for epoch in range(self.epoch):
            Train_data, Validation_data, Test_data = \
                data.construct_data(self.list_size, self.features_num, self.batch_size)
            t1 = time.time()
            for batch_xs in Train_data:
                # Fit training
                loss = self.partial_fit(batch_xs)
            t2 = time.time()

            # output validation
            valid_result = self.evaluate(Validation_data)
            test_result = self.evaluate(Test_data)

            self.valid_ndcg.append(valid_result)
            self.test_ndcg.append(test_result)
            if self.verbose > 0 and epoch % self.verbose == 0:
                print("Epoch %d [%.1f s]\tvalidation=%.4f, test=%.4f [%.1f s]"
                      % (epoch + 1, t2 - t1, valid_result, test_result, time.time() - t2))
            if self.eva_termination(self.valid_ndcg):
                break
            # 保存embedding
            self.get_items_embedding(data.dict_id2bid, epoch)
        self.saver.save(self.sess, self.save_file)

    def eva_termination(self, valid):
        # 早停
        if len(valid) > 5:
            if valid[-1] < valid[-2] and valid[-2] < valid[-3] and valid[-3] < valid[-4] and valid[-4] < valid[-5]:
                return True
        return False

    def evaluate(self, data):  # evaluate the results for an input set
        total_ndcg = total_nums = step = 0
        for batch_xs in data:
            feed_dict = {self.train_features: batch_xs['X'], self.train_labels: batch_xs['Y'],
                         self.train_values: batch_xs['X_value'], self.dropout_keep: 1.0, self.train_phase: False}
            predictions = self.sess.run(self.logit, feed_dict=feed_dict)
            ndcg, nums = cal_ndcg(batch_xs['Y'], predictions, self.list_size)
            total_ndcg += ndcg
            total_nums += nums
            step += 1
            if step % 10000 == 0:
                print(f'eval step: {step}')
        return total_ndcg / total_nums

    def get_items_embedding(self, dict_id2bid, epoch):
        items_embedding = self.sess.run(self.weights['feature_embeddings'])
        items_embedding = pd.DataFrame(items_embedding)
        items_embedding.index = items_embedding.index.map(lambda x: dict_id2bid[x])
        items_embedding.to_csv(f'fm_bid_embedding_{epoch}.txt', sep=',', header=None)


if __name__ == '__main__':
    # Data loading
    args = parse_args()
    data = LoadData(args.path, args.dataset, args.list_size, args.features_num, args.batch_size, args.pretrain)

    if args.verbose > 0:
        print(f"FM: dataset={args.dataset}, factors={args.hidden_factor}, "
              f"#epoch={args.epoch}, batch={args.batch_size}, lr={args.lr}, lambda={args.lamda}, "
              f"keep={args.keep_prob}, optimizer={args.optimizer}, batch_norm={args.batch_norm}")

    save_file = f'save/{args.dataset}_{args.hidden_factor}'
    # Training
    t1 = time.time()
    # feature2id 特征转id字典， pretrain：是否加载预训练向量， save_file：模型保存文件名字
    # hidden_factor： 向量长度， epoch： 迭代次数， batch_size： batch大小（每条对应一个group）
    # lr： 学习率， lamda： 正则化系数， keep_prob：dropout参数， keep_prob： 优化器，
    # batch_norm： 是否增加batch_normal, verbose: 测试flag, log_dir: 日志文件夹， pretrain_embedding：预训练词向量
    model = LambdaFM(args.feature_id_num, args.pretrain, save_file, args.hidden_factor, args.epoch, args.batch_size,
                     args.lr, args.lamda, args.keep_prob, args.optimizer, args.batch_norm, args.verbose,
                     args.log_dir, data.pretrain_embedding, args.list_size, args.features_num)
    model.train(data)
    # model.get_items_embedding()
    # Find the best validation result across iterations
    best_valid_score = max(model.valid_ndcg)
    best_epoch = model.valid_ndcg.index(best_valid_score)
    print("Best Iter(validation)= %d\t valid = %.4f, test = %.4f [%.1f s]"
          % (best_epoch + 1, model.valid_ndcg[best_epoch], model.test_ndcg[best_epoch],
             time.time() - t1))
