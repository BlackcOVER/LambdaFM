import math
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from time import time
import argparse
from load_data import LoadData
from loss import create_ndcg_lambda_weight, PairwiseLogisticLoss


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run FM.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='test',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=10,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=-1,
                        help='flag for pretrain. 1: initialize from pretrain; 0: '
                             'randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=32,
                        help='Number of hidden factors.')
    parser.add_argument('--cate_hidden_factor', type=int, default=32,
                        help='Number of cate hidden factors.')  # 暂时设置成同一维度
    parser.add_argument('--lamda', type=float, default=0,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--keep_prob', type=float, default=0.5,
                        help='Keep probility (1-dropout_ratio) for the Bi-Interaction layer. 1: no dropout')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='square_loss',
                        help='Specify a loss type (square_loss or log_loss).')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, '
                             'AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=1,
                        help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--log_dir', type=str, default='log',
                        help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--hist_nums', type=int, default=20,
                        help='max nums of features')

    return parser.parse_args()


class FM(BaseEstimator, TransformerMixin):
    def __init__(self, features_M, features_C, pretrain_flag, save_file, hidden_factor, loss_type, epoch, batch_size,
                 learning_rate, lambda_bilinear, keep, optimizer_type, batch_norm, verbose, log_dir, hist_nums, pretrain_embedding,
                 params, rank=True, random_seed=2019):
        # bind params to class
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_factor = hidden_factor
        self.save_file = save_file
        self.pretrain_flag = pretrain_flag
        self.loss_type = loss_type
        self.features_M = features_M
        self.features_C = features_C
        self.params = params
        self.lambda_bilinear = lambda_bilinear
        self.keep = keep
        self.epoch = epoch
        self.log_dir = log_dir
        self.hist_nums = hist_nums
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.verbose = verbose
        # performance of each epoch
        self.train_rmse, self.valid_rmse, self.test_rmse = list(), list(), list()
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
            self.train_features = tf.placeholder(tf.int32, shape=[None, None])  # None * features_M
            # scores
            self.train_values = tf.placeholder(tf.float32, shape=[None, None])  # None * features_M
            self.train_labels = tf.placeholder(tf.float32, shape=[None, 1])  # None * 1
            # sparse cate
            self.train_c1_features = tf.sparse_placeholder(tf.int32, name='c1')
            self.train_c2_features = tf.sparse_placeholder(tf.int32, name='c2')
            self.train_c3_features = tf.sparse_placeholder(tf.int32, name='c3')

            self.dropout_keep = tf.placeholder(tf.float32)
            self.train_phase = tf.placeholder(tf.bool)

            # Variables.
            self.weights = self._initialize_weights()

            # Model.
            # _________ sum_square part _____________
            # get the summed up embeddings of features.
            nonzero_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features)
            c1_embeddings = \
                tf.nn.embedding_lookup_sparse(self.weights['cate_embeddings'], self.train_c1_features, None, combiner='mean')
            c2_embeddings = \
                tf.nn.embedding_lookup_sparse(self.weights['cate_embeddings'], self.train_c2_features, None, combiner='mean')
            c3_embeddings = \
                tf.nn.embedding_lookup_sparse(self.weights['cate_embeddings'], self.train_c3_features, None, combiner='mean')

            c1_embeddings = tf.reshape(c1_embeddings, shape=[self.batch_size, self.hist_nums, self.hidden_factor])
            c2_embeddings = tf.reshape(c2_embeddings, shape=[self.batch_size, self.hist_nums, self.hidden_factor])
            c3_embeddings = tf.reshape(c3_embeddings, shape=[self.batch_size, self.hist_nums, self.hidden_factor])

            self.nonzero_embeddings = \
                tf.concat([nonzero_embeddings, c1_embeddings, c2_embeddings, c3_embeddings], axis=-1)
            train_values = tf.reshape(self.train_values,
                                      [tf.shape(self.train_features)[0], tf.shape(self.train_features)[1], 1])
            value_matrix = tf.tile(train_values, [1, 1, tf.shape(self.nonzero_embeddings)[2]])
            value_embedding = tf.multiply(self.nonzero_embeddings, value_matrix)

            self.summed_features_emb = tf.reduce_sum(value_embedding, 1)  # None * K
            # get the element-multiplication
            self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

            # _________ square_sum part _____________
            self.squared_features_emb = tf.square(value_embedding)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

            # ________ FM __________
            self.FM = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)  # None * K
            if self.batch_norm:
                self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase)
            self.FM = tf.nn.dropout(self.FM, self.dropout_keep)  # dropout at the FM layer

            # _________out _________
            Bilinear = tf.reduce_sum(self.FM, 1, keep_dims=True)  # None * 1
            self.Feature_bias = tf.reduce_sum(
                tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_features), 1)  # None * 1
            Bias = self.weights['bias'] * tf.ones_like(self.train_labels)  # None * 1
            self.out = tf.add_n([Bilinear, self.Feature_bias, Bias])  # None * 1

            # Compute the loss.
            l2_regularizer = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

            if self.loss_type == 'square_loss':
                if self.lambda_bilinear > 0:
                    self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out)) + tf.add_n(l2_regularizer)
                else:
                    self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out))
            elif self.loss_type == 'log_loss':
                self.out = tf.sigmoid(self.out)
                if self.lambda_bilinear > 0:
                    self.loss = tf.losses.log_loss(self.train_labels, self.out,
                                                   weight=1.0, epsilon=1e-07) + tf.add_n(l2_regularizer)  # regulizer
                else:
                    self.loss = tf.losses.log_loss(self.train_labels, self.out, weight=1.0, epsilon=1e-07)
            elif self.loss_type == 'pairwise_loss':
                self.pairwise_loss_func = PairwiseLogisticLoss('fm_loss', self.lambda_weight)
                self.loss = self.pairwise_loss_func.compute(self.train_labels, self.out, None, None)
            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8)
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
                print(f"#params: {total_parameters}")

    def _initialize_weights(self):
        all_weights = dict()
        # 控制是否预加载
        if self.pretrain_flag > 0:
            feature_embeddings = np.random.normal(0.0, 0.01, [self.features_M, self.hidden_factor])
            for index, vec in self.pretrain_embedding.items():
                feature_embeddings[index] = vec
            all_weights['feature_embeddings'] = tf.Variable(feature_embeddings, dtype=tf.float32)
        else:
            all_weights['feature_embeddings'] = \
                tf.get_variable(name='feature_embeddings', shape=[self.features_M, self.hidden_factor],
                                initializer=tf.random_normal_initializer(0.0, 0.01),
                                regularizer=tf.keras.regularizers.l2(self.lambda_bilinear))  # features_M * K
        all_weights['feature_bias'] = \
            tf.get_variable(name='feature_bias', shape=[self.features_M, 1],
                            initializer=tf.random_normal_initializer(0.0, 0.0))  # features_M * 1
        all_weights['bias'] = tf.get_variable(name='bias', initializer=tf.constant(0.0))  # 1 * 1
        all_weights['cate_embeddings'] = \
            tf.get_variable(name='cate_embeddings', shape=[self.features_C, self.hidden_factor],
                            initializer=tf.random_normal_initializer(0.0, 0.01),
                            regularizer=tf.keras.regularizers.l2(self.lambda_bilinear))  # features_M * K
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
                     self.train_values: data['X_value'], self.train_c1_features: data['X_cate']['c1'],
                     self.train_c2_features: data['X_cate']['c2'], self.train_c3_features: data['X_cate']['c3'],
                     self.dropout_keep: self.keep, self.train_phase: True}

        loss, opt = self.sess.run((self.loss, self.train_op), feed_dict=feed_dict)
        return loss

    def get_random_block_from_data(self, data, batch_size, step, params):  # generate a random block of training data
        # 每次随机生成batch
        # start_index = np.random.randint(0, len(data['Y']) - batch_size)
        start_index = step * batch_size
        X, Y, X_value, X_cate = list(), list(), list(), \
                                {'c1': {'indices': list(), 'values': list()},
                                 'c2': {'indices': list(), 'values': list()},
                                 'c3': {'indices': list(), 'values': list()}}
        # forward get sample
        i = start_index
        count = 0
        # 针对X中每一条样本，有k条候选书籍
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                X_value.append(data['X_value'][i])
                # 每一条样本中的每一本书籍
                for j in range(len(data['X_cate'][i])):
                    # 每一本书的每一个分类j
                    for cate_level in ['c1', 'c2', 'c3']:
                        # 每个
                        for k in range(len(data['X_cate'][i][j][cate_level])):
                            X_cate[cate_level]['indices'].append([count * (params['hist_nums']) + j, k])
                            X_cate[cate_level]['values'].append(data['X_cate'][i][j][cate_level][k])
                i = i + 1
                count += 1
            else:
                break
        X_cate['c1'] = tf.SparseTensorValue(X_cate['c1']['indices'], X_cate['c1']['values'],
                                            dense_shape=[batch_size * len(data['X'][0]), params['max_c1_nums']])
        X_cate['c2'] = tf.SparseTensorValue(X_cate['c2']['indices'], X_cate['c2']['values'],
                                            dense_shape=[batch_size * len(data['X'][0]), params['max_c2_nums']])
        X_cate['c3'] = tf.SparseTensorValue(X_cate['c3']['indices'], X_cate['c3']['values'],
                                            dense_shape=[batch_size * len(data['X'][0]), params['max_c3_nums']])

        return {'X': X, 'Y': Y, 'X_value': X_value, 'X_cate': X_cate}

    def shuffle_in_unison_scary(self, a, b, c, d):  # shuffle two lists simutaneously
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        np.random.set_state(rng_state)
        np.random.shuffle(d)

    def train(self, Train_data, Validation_data, Test_data, params):  # fit a dataset
        # Check Init performance
        if self.verbose > 0:
            t2 = time()
            init_train = self.evaluate(Train_data)
            init_valid = self.evaluate(Validation_data)
            init_test = self.evaluate(Test_data)
            print("Init: \t train=%.4f, validation=%.4f, test=%.4f [%.1f s]" % (
            init_train, init_valid, init_test, time() - t2))
        # TODO 生成器
        for epoch in range(self.epoch):
            t1 = time()
            # TODO 在原始文件进行shuffle
            self.shuffle_in_unison_scary(Train_data['X'], Train_data['Y'], Train_data['X_value'], Train_data['X_cate'])
            # TODO  总样本数量设置为超参
            total_batch = int(len(Train_data['Y']) / self.batch_size)
            for i in range(total_batch):
                # generate a batch
                batch_xs = self.get_random_block_from_data(Train_data, self.batch_size, i, params)
                # Fit training
                loss = self.partial_fit(batch_xs)
            t2 = time()

            # output validation
            train_result = self.evaluate(Train_data)
            valid_result = self.evaluate(Validation_data)
            test_result = self.evaluate(Test_data)

            self.train_rmse.append(train_result)
            self.valid_rmse.append(valid_result)
            self.test_rmse.append(test_result)
            if self.verbose > 0 and epoch % self.verbose == 0:
                print("Epoch %d [%.1f s]\ttrain=%.4f, validation=%.4f, test=%.4f [%.1f s]"
                      % (epoch + 1, t2 - t1, train_result, valid_result, test_result, time() - t2))
            if self.eva_termination(self.valid_rmse):
                break

        if self.pretrain_flag < 0:
            print("Save model to file as pretrain.")
            self.saver.save(self.sess, self.save_file)

    def eva_termination(self, valid):
        if self.loss_type == 'square_loss':
            if len(valid) > 5:
                if valid[-1] > valid[-2] and valid[-2] > valid[-3] and valid[-3] > valid[-4] and valid[-4] > valid[-5]:
                    return True
        else:
            if len(valid) > 5:
                if valid[-1] < valid[-2] and valid[-2] < valid[-3] and valid[-3] < valid[-4] and valid[-4] < valid[-5]:
                    return True
        return False

    def evaluate(self, data):  # evaluate the results for an input set
        num_example = len(data['Y'])
        feed_dict = {self.train_features: data['X'], self.train_labels: [[y] for y in data['Y']],
                     self.train_values: data['X_value'],
                     self.dropout_keep: 1.0, self.train_phase: False}
        predictions = self.sess.run((self.out), feed_dict=feed_dict)
        y_pred = np.reshape(predictions, (num_example,))
        y_true = np.reshape(data['Y'], (num_example,))
        if self.loss_type == 'square_loss':
            predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
            predictions_bounded = np.minimum(predictions_bounded,
                                             np.ones(num_example) * max(y_true))  # bound the higher values
            RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
            return RMSE
        elif self.loss_type == 'log_loss':
            logloss = log_loss(y_true, y_pred)  # I haven't checked the log_loss
            return logloss

    def get_items_embedding(self):
        items_embedding = self.sess.run(self.weights['feature_embeddings'])
        print(type(items_embedding))
        print(items_embedding.shape)


if __name__ == '__main__':
    # Data loading
    args = parse_args()
    data = LoadData(args.path, args.dataset, args.loss_type, args.hist_nums)








    if args.verbose > 0:
        print(f"FM: dataset={args.dataset}, factors={args.hidden_factor}, loss_type={args.loss_type}, "
              f"#epoch={args.epoch}, batch={args.batch_size}, lr={args.lr}, lambda={args.lamda}, "
              f"keep={args.keep_prob}, optimizer={args.optimizer}, batch_norm={args.batch_norm}")

    save_file = f'pretrain/{args.dataset}_{args.hidden_factor}/{args.dataset}_{args.hidden_factor}'
    # Training
    t1 = time()
    params = {'max_c1_nums': data.max_c1_nums, 'max_c2_nums': data.max_c2_nums,
              'max_c3_nums': data.max_c3_nums, 'hist_nums': args.hist_nums}
    model = FM(data.features_M, data.features_C, args.pretrain, save_file, args.hidden_factor,
               args.loss_type, args.epoch, args.batch_size, args.lr, args.lamda,
               args.keep_prob, args.optimizer, args.batch_norm, args.verbose, args.log_dir,
               args.hist_nums, data.pretrain_embedding, params)
    model.train(data.Train_data, data.Validation_data, data.Test_data, params)
    # model.get_items_embedding()
    # Find the best validation result across iterations
    best_valid_score = 0
    if args.loss_type == 'square_loss':
        best_valid_score = min(model.valid_rmse)
    elif args.loss_type == 'log_loss':
        best_valid_score = max(model.valid_rmse)
    best_epoch = model.valid_rmse.index(best_valid_score)
    print("Best Iter(validation)= %d\t train = %.4f, valid = %.4f, test = %.4f [%.1f s]"
          % (best_epoch + 1, model.train_rmse[best_epoch], model.valid_rmse[best_epoch], model.test_rmse[best_epoch],
             time() - t1))
