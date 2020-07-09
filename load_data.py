import json
import six

import tensorflow as tf
import numpy as np


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialize data iterator after session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_fn = None

    def after_create_session(self, session, coord):
        """Initialize the iterator after the session has been created."""
        del coord
        self.iterator_initializer_fn(session)


def get_train_inputs(features_id, features_value, labels, batch_size):
    """Set up training input in batches."""
    iterator_initializer_hook = IteratorInitializerHook()

    def _train_input_fn():
        """Defines training input fn."""
        features_id_placeholder = tf.placeholder(np.int32, shape=features_id.shape)
        features_value_placeholder = tf.placeholder(np.int32, shape=features_value.shape)
        labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
        dataset = tf.data.Dataset.from_tensor_slices(
            (features_id_placeholder, features_value_placeholder, labels_placeholder))
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        feed_dict = {labels_placeholder: labels}
        feed_dict.update(
            {features_id_placeholder: features_id, features_value_placeholder: features_value})
        iterator_initializer_hook.iterator_initializer_fn = (
            lambda sess: sess.run(iterator.initializer, feed_dict=feed_dict))
        return iterator.get_next()

    return _train_input_fn, iterator_initializer_hook


def load_libsvm_data(path, list_size, features_num, batch_size):
    '''
    :param path:
    :param list_size: group限制大小
    :param features_num: 特征限制大小
    :return:
    '''
    """Returns features and labels in numpy.array."""

    def _parse_line(line, features_num):
        """Parses a single line in LibSVM format."""
        labels, qid, features = line.split("#")
        labels = [kv.split(':') for kv in labels.split('\t')]
        labels = {int(k): int(v) for k, v in labels}
        kv_pairs = [kv.split(":") for kv in features.split('\t')]
        features_id = [int(k) for k, v in kv_pairs]
        features_value = [float(v) for k, v in kv_pairs]
        real_features_num = len(features_id)
        pad_num = features_num - real_features_num - 1
        if pad_num > 0:
            features_id += [0] * pad_num
            features_value += [0] * pad_num
        return qid, features_id, features_value, labels

    tf.logging.info("Loading data from {}".format(path))

    # The 0-based index assigned to a query.
    # query对应的index
    qid_to_index = {}
    # The number of docs seen so far for a query.
    # 每个query 有多少条样本
    qid_to_ndoc = {}
    # Each feature is mapped an array with [num_queries, list_size, 1]. Label has
    # a shape of [num_queries, list_size]. We use list for each of them due to the
    # unknown number of quries.
    # TODO
    feature_id_map = list()
    feature_value_map = list()
    label_list = []
    total_docs = 0
    discarded_docs = 0
    with open(path, "rt") as f:
        batch_count = 0
        for line in f:
            qid, features_id, features_value, labels = _parse_line(line, features_num)
            batch_count += 1
            # Create index and allocate space for a new query.
            qid_to_index[qid] = len(qid_to_index)
            qid_to_ndoc[qid] = 0
            feature_id_map.append(np.zeros([list_size, features_num], dtype=np.int32))
            feature_value_map.append(np.zeros([list_size, features_num], dtype=np.float32))
            label_list.append(np.ones([list_size], dtype=np.float32) * -1.)
            for key, label in labels.items():
                doc_idx = qid_to_ndoc[qid]
                qid_to_ndoc[qid] += 1
                if doc_idx >= list_size:
                    discarded_docs += 1
                    continue
                feature_id_map[-1][doc_idx] = np.array(features_id + [key], dtype=np.int32)
                feature_value_map[-1][doc_idx] = np.array(features_value + [1], dtype=np.float32)
                label_list[-1][doc_idx] = label
            total_docs += len(labels)
            if batch_count == batch_size:
                feed_dict = construct_dataset(feature_id_map, np.array(label_list), feature_value_map)
                yield feed_dict
                batch_count = 0
                feature_id_map = list()
                feature_value_map = list()
                label_list = []
        if 0 < batch_count < batch_size:
            feed_dict = construct_dataset(feature_id_map, np.array(label_list), feature_value_map)
            yield feed_dict


def construct_dataset(X_, Y_, X_value):
        Data_Dic = dict()
        # 根据样本特征长度从大到小排列
        Data_Dic['Y'] = Y_
        Data_Dic['X'] = X_
        Data_Dic['X_value'] = X_value
        return Data_Dic


class LoadData(object):
    '''given the path of data, return the data format for DeepFM
    :param path
    return:
    Train_data: a dictionary, 'Y' refers to a list of y values;
                'X' refers to a list of features_M dimension vectors with 0 or 1 entries
    Test_data: same as Train_data
    Validation_data: same as Train_data
    '''

    # Three files are needed in the path
    def __init__(self, path, dataset, list_size, features_num, batch_size, pretrain_flag=False):
        # 数据路径
        self.path = path + dataset + "/"
        # 训练集和
        self.trainfile = self.path + "train.txt"
        # 测试集合
        self.testfile = self.path + "test.txt"
        # 验证集合
        self.validationfile = self.path + "val.txt"
        # 数据集
        self.dict_map_item2id_path = self.path + "bid_id_map.csv"
        self.load_dict_bid2id(self.dict_map_item2id_path)
        if pretrain_flag:
            self.pretrain_path = self.path + "pretrain_embedding.csv"
            self.construct_pretrain()
        else:
            self.pretrain_embedding = None

    def load_dict_bid2id(self, path):
        self.dict_bid2id = dict()
        self.dict_id2bid = dict()
        with open(path) as file:
            for line in file:
                bid, index = line.strip('\n').split('\t')
                self.dict_bid2id[bid] = int(index)
                self.dict_id2bid[int(index)] = bid

    def construct_pretrain(self):

        self.pretrain_embedding = dict()
        with open(self.pretrain_path) as file:
            for line in file:
                items = line.strip('\n').split(',')
                word, vec = items[0], items[1:]
                if word not in self.dict_bid2id:
                    continue
                else:
                    self.pretrain_embedding[self.dict_bid2id[word]] = np.array([float(x) for x in vec], dtype=np.float32)

    def construct_data(self, list_size, features_num, batch_size):
        Train_data = load_libsvm_data(self.trainfile, list_size, features_num, batch_size)

        Validation_data = load_libsvm_data(self.validationfile, list_size, features_num, batch_size)
        Test_data = load_libsvm_data(self.testfile, list_size, features_num, batch_size)

        return Train_data, Validation_data, Test_data
