#  Copyright (c) 2021, The SmbRec Authors.  All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import pandas as pd
import datetime
import itertools
import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.keras.initializers import Zeros, glorot_normal,glorot_uniform, TruncatedNormal
from collections import namedtuple, OrderedDict

########################################################################
               #################数据预处理##############
########################################################################


# 定义参数类型
SparseFeat = namedtuple('SparseFeat', ['name', 'voc_size', 'hash_size', 'vocab', 'share_embed', 'embed_dim', 'dtype'])
DenseFeat = namedtuple('DenseFeat', ['name', 'pre_embed', 'reduce_type', 'dim', 'dtype'])
VarLenSparseFeat = namedtuple('VarLenSparseFeat',
                              ['name', 'voc_size', 'hash_size', 'vocab', 'share_embed', 'weight_name', 'combiner',
                               'embed_dim', 'maxlen', 'dtype'])

# 筛选有效
valid_keyword = pd.read_csv('/opt/data1/xululu/keyword_freq.csv', sep='\t')
valid_keyword = valid_keyword[valid_keyword.cnt >= 2]

# 筛选实体标签categorical
CATEGORICAL_MAP = {
    "keyword": valid_keyword.keyword_tag.unique().tolist(),
    "dislike_keyword": valid_keyword.keyword_tag.unique().tolist(),
    "most_topic": list(range(0, 710)),
    "dislike_topic": list(range(0, 710)),
}

feature_columns = [
    DenseFeat(name='c_topic_id_ctr', pre_embed=None, reduce_type=None, dim=1, dtype="float32"),
    # SparseFeat(name="user_id", voc_size=1000000, hash_size= 1000000,  vocab=None, share_embed=None, embed_dim=16, dtype='string'),
    SparseFeat(name="c_follow_topic_id", voc_size=2, hash_size=None, vocab=None, share_embed=None, embed_dim=8,
               dtype='int32'),
    SparseFeat(name="c_search_keyword", voc_size=2, hash_size=None, vocab=None, share_embed=None, embed_dim=8,
               dtype='int32'),
    SparseFeat(name="exposure_hourdiff", voc_size=6, hash_size=None, vocab=None, share_embed=None, embed_dim=8,
               dtype='int32'),
    SparseFeat(name="reply", voc_size=6, hash_size=None, vocab=None, share_embed=None, embed_dim=8, dtype='int32'),
    # SparseFeat(name="share", voc_size=6, hash_size= None, vocab=None,share_embed=None, embed_dim=8, dtype='int32'),
    SparseFeat(name="recommend", voc_size=6, hash_size=None, vocab=None, share_embed=None, embed_dim=8, dtype='int32'),
    SparseFeat(name='topic_id', voc_size=720, hash_size=None, vocab=None, share_embed=None, embed_dim=8, dtype='int32'),
    SparseFeat(name='exposure_hour', voc_size=25, hash_size=None, vocab=None, share_embed=None, embed_dim=8,
               dtype='int32'),
    VarLenSparseFeat(name="follow_topic_id", voc_size=720, hash_size=None, vocab=None, share_embed='topic_id',
                     weight_name=None, combiner='sum', embed_dim=8, maxlen=20, dtype='int32'),
    VarLenSparseFeat(name="major_topic", voc_size=720, hash_size=None, vocab=None, share_embed='topic_id',
                     weight_name=None, combiner='sum', embed_dim=8, maxlen=10, dtype='int32'),
    VarLenSparseFeat(name="keyword", voc_size=20000, hash_size=None, vocab='keyword', share_embed=None,
                     weight_name=None, combiner='sum', embed_dim=8, maxlen=5, dtype='int32'),
    VarLenSparseFeat(name="search_keyword", voc_size=20000, hash_size=None, vocab='keyword', share_embed='keyword',
                     weight_name=None, combiner='sum', embed_dim=12, maxlen=5, dtype='int32'),
    VarLenSparseFeat(name="major_keyword", voc_size=20000, hash_size=None, vocab='keyword', share_embed='keyword',
                     weight_name=None, combiner='sum', embed_dim=8, maxlen=30, dtype='int32'),
    VarLenSparseFeat(name="topic_dislike_7d", voc_size=720, hash_size=None, vocab='dislike_topic',
                     share_embed='dislike_topic', weight_name=None, combiner='sum', embed_dim=8, maxlen=7,
                     dtype='int32'),

]

# 用户特征及贴子特征
dnn_feature_columns_name = [
    'c_follow_topic_id', 'c_search_keyword', 'c_topic_id_ctr',
    'c_major_topic_id', 'c_major_keyword', 'c_topic_dislike_7d',
    "topic_id", 'exposure_hour', "exposure_hourdiff", 'reply', 'recommend', 'keyword', "entity",
    'follow_topic_id', "search_keyword",
    'major_keyword', 'major_topic', 'topic_dislike_7d',
]

dnn_feature_columns = [col for col in feature_columns if col.name in dnn_feature_columns_name]

# 离散分桶边界定义
BUCKET_DICT = {
    'exposure_hourdiff': [3, 7, 15, 33],
    'reply': [12, 30, 63, 136],
    #     'share': [2, 11],
    'recommend': [1, 6, 16, 45],
}

DEFAULT_VALUES = [
    ['0'], [0], [0], [0.0], [0],
    [0], [0.0], [0], [0], [0],
    [0], [0], [0], [0], [0],
    ['0'], ['0'], ['0'], ['0'], ['0'],
    ['0'], ['0'],

]

COL_NAME = ['user_id', 'post_id', 'label', 'dur', 'c_follow_topic_id',
            'c_search_keyword', 'c_topic_id_ctr', 'c_major_topic_id', 'c_major_keyword', 'c_topic_dislike_7d',
            'topic_id', 'exposure_hour', 'exposure_hourdiff', 'reply', 'recommend',
            'keyword', 'entity', 'follow_topic_id', 'search_keyword', 'major_topic',
            'major_keyword', 'topic_dislike_7d']


# 特征解析
def _parse_function(example_proto):
    item_feats = tf.io.decode_csv(example_proto, record_defaults=DEFAULT_VALUES, field_delim='\t')
    parsed = dict(zip(COL_NAME, item_feats))

    feature_dict = {}
    for feat_col in feature_columns:
        if isinstance(feat_col, VarLenSparseFeat):
            if feat_col.weight_name is not None:
                kvpairs = tf.strings.split([parsed[feat_col.name]], ',').values[:feat_col.maxlen]
                kvpairs = tf.strings.split(kvpairs, ':')
                kvpairs = kvpairs.to_tensor()
                feat_ids, feat_vals = tf.split(kvpairs, num_or_size_splits=2, axis=1)
                feat_ids = tf.reshape(feat_ids, shape=[-1])
                feat_vals = tf.reshape(feat_vals, shape=[-1])
                if feat_col.dtype != 'string':
                    feat_ids = tf.strings.to_number(feat_ids, out_type=tf.int32)
                feat_vals = tf.strings.to_number(feat_vals, out_type=tf.float32)
                feature_dict[feat_col.name] = feat_ids
                feature_dict[feat_col.weight_name] = feat_vals
            else:
                feat_ids = tf.strings.split([parsed[feat_col.name]], ',').values[:feat_col.maxlen]
                feat_ids = tf.reshape(feat_ids, shape=[-1])
                if feat_col.dtype != 'string':
                    feat_ids = tf.strings.to_number(feat_ids, out_type=tf.int32)
                feature_dict[feat_col.name] = feat_ids

        elif isinstance(feat_col, SparseFeat):
            feature_dict[feat_col.name] = parsed[feat_col.name]

        elif isinstance(feat_col, DenseFeat):
            if not feat_col.pre_embed:
                feature_dict[feat_col.name] = parsed[feat_col.name]
            elif feat_col.reduce_type is not None:
                keys = tf.strings.split(parsed[feat_col.pre_embed], ',')
                emb = tf.nn.embedding_lookup(params=ITEM_EMBEDDING, ids=ITEM_ID2IDX.lookup(keys))
                emb = tf.reduce_mean(emb, axis=0) if feat_col.reduce_type == 'mean' else tf.reduce_sum(emb, axis=0)
                feature_dict[feat_col.name] = emb
            else:
                emb = tf.nn.embedding_lookup(params=ITEM_EMBEDDING, ids=ITEM_ID2IDX.lookup(parsed[feat_col.pre_embed]))
                feature_dict[feat_col.name] = emb
        else:
            raise "unknown feature_columns...."

    # 分桶离散化
    for ft in BUCKET_DICT:
        feature_dict[ft] = tf.raw_ops.Bucketize(
            input=feature_dict[ft],
            boundaries=BUCKET_DICT[ft])

    label = parsed['label']
    duration = parsed['dur']

    return feature_dict, (label, duration)


pad_shapes = {}
pad_values = {}

for feat_col in feature_columns:
    if isinstance(feat_col, VarLenSparseFeat):
        max_tokens = feat_col.maxlen
        pad_shapes[feat_col.name] = tf.TensorShape([max_tokens])
        pad_values[feat_col.name] = '-1' if feat_col.dtype == 'string' else -1

        if feat_col.weight_name is not None:
            pad_shapes[feat_col.weight_name] = tf.TensorShape([max_tokens])
            pad_values[feat_col.weight_name] = tf.constant(-1, dtype=tf.float32)

    # no need to pad labels
    elif isinstance(feat_col, SparseFeat):
        pad_values[feat_col.name] = '-1' if feat_col.dtype == 'string' else -1
        pad_shapes[feat_col.name] = tf.TensorShape([])
    elif isinstance(feat_col, DenseFeat):
        if not feat_col.pre_embed:
            pad_shapes[feat_col.name] = tf.TensorShape([])
        else:
            pad_shapes[feat_col.name] = tf.TensorShape([feat_col.dim])
        pad_values[feat_col.name] = 0.0

pad_shapes = (pad_shapes, (tf.TensorShape([]), tf.TensorShape([])))
pad_values = (pad_values, (tf.constant(0, dtype=tf.int32), tf.constant(0.0, dtype=tf.float32)))

# 训练数据
filenames = tf.data.Dataset.list_files([

    './test_data.tsv',
])
dataset = filenames.flat_map(
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1))

batch_size = 100
dataset = dataset.map(_parse_function, num_parallel_calls=50)
dataset = dataset.repeat()
dataset = dataset.shuffle(buffer_size=batch_size)  # 在缓冲区中随机打乱数据
dataset = dataset.padded_batch(batch_size=batch_size,
                               padded_shapes=pad_shapes,
                               padding_values=pad_values)  # 每1024条数据为一个batch，生成一个新的Datasets
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# 验证集
filenames_val = tf.data.Dataset.list_files(['./test_data.tsv'])
dataset_val = filenames_val.flat_map(
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1))

val_batch_size = 100
dataset_val = dataset_val.map(_parse_function, num_parallel_calls=50)
dataset_val = dataset_val.padded_batch(batch_size=val_batch_size,
                                       padded_shapes=pad_shapes,
                                       padding_values=pad_values)  # 每1024条数据为一个batch，生成一个新的Datasets
dataset_val = dataset_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


########################################################################
               #################自定义Layer##############
########################################################################

# 离散特征查找表层
class VocabLayer(Layer):
    def __init__(self, keys, mask_value=None, **kwargs):
        super(VocabLayer, self).__init__(**kwargs)
        self.mask_value = mask_value
        vals = tf.range(2, len(keys) + 2)
        vals = tf.constant(vals, dtype=tf.int32)
        keys = tf.constant(keys)
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, vals), 1)

    def call(self, inputs):
        idx = self.table.lookup(inputs)
        if self.mask_value is not None:
            masks = tf.not_equal(inputs, self.mask_value)
            paddings = tf.ones_like(idx) * (-1)  # mask成 -1
            idx = tf.where(masks, idx, paddings)
        return idx

    def get_config(self):
        config = super(VocabLayer, self).get_config()
        config.update({'mask_value': self.mask_value, })
        return config


# multi-hot 特征EmbeddingLookup层
class EmbeddingLookupSparse(Layer):
    def __init__(self, embedding, has_weight=False, combiner='sum', **kwargs):

        super(EmbeddingLookupSparse, self).__init__(**kwargs)
        self.has_weight = has_weight
        self.combiner = combiner
        self.embedding = embedding

    def build(self, input_shape):
        super(EmbeddingLookupSparse, self).build(input_shape)

    def call(self, inputs):
        if self.has_weight:
            idx, val = inputs
            combiner_embed = tf.nn.embedding_lookup_sparse(self.embedding, sp_ids=idx, sp_weights=val,
                                                           combiner=self.combiner)
        else:
            idx = inputs
            combiner_embed = tf.nn.embedding_lookup_sparse(self.embedding, sp_ids=idx, sp_weights=None,
                                                           combiner=self.combiner)
        return tf.expand_dims(combiner_embed, 1)

    def get_config(self):
        config = super(EmbeddingLookupSparse, self).get_config()
        config.update({'has_weight': self.has_weight, 'combiner': self.combiner})
        return config


# 单值离散特征EmbeddingLookup 层
class EmbeddingLookup(Layer):
    def __init__(self, embedding, **kwargs):
        super(EmbeddingLookup, self).__init__(**kwargs)
        self.embedding = embedding

    def build(self, input_shape):
        super(EmbeddingLookup, self).build(input_shape)

    def call(self, inputs):
        idx = tf.cast(inputs, tf.int32)
        embed = tf.nn.embedding_lookup(params=self.embedding, ids=idx)
        return embed

    def get_config(self):
        config = super(EmbeddingLookup, self).get_config()
        return config


# 稠密转稀疏
class DenseToSparseTensor(Layer):
    def __init__(self, mask_value=-1, **kwargs):
        super(DenseToSparseTensor, self).__init__()
        self.mask_value = mask_value

    def call(self, dense_tensor):
        idx = tf.where(tf.not_equal(dense_tensor, tf.constant(self.mask_value, dtype=dense_tensor.dtype)))
        sparse_tensor = tf.SparseTensor(idx, tf.gather_nd(dense_tensor, idx), tf.shape(dense_tensor, out_type=tf.int64))
        return sparse_tensor

    def get_config(self):
        config = super(DenseToSparseTensor, self).get_config()
        config.update({'mask_value': self.mask_value})
        return config


# 自定义hash层
class HashLayer(Layer):
    """
    hash the input to [0,num_buckets)
    if mask_zero = True,0 or 0.0 will be set to 0,other value will be set in range[1,num_buckets)
    """

    def __init__(self, num_buckets, mask_zero=False, **kwargs):
        self.num_buckets = num_buckets
        self.mask_zero = mask_zero
        super(HashLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(HashLayer, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        if x.dtype != tf.string:
            zero = tf.as_string(tf.ones([1], dtype=x.dtype) * (-1))
            x = tf.as_string(x, )
        else:
            zero = tf.as_string(tf.ones([1], dtype=x.dtype) * (-1))

        num_buckets = self.num_buckets if not self.mask_zero else self.num_buckets - 1
        hash_x = tf.strings.to_hash_bucket_fast(x, num_buckets, name=None)
        if self.mask_zero:
            #             mask = tf.cast(tf.not_equal(x, zero), dtype='int64')
            masks = tf.not_equal(x, zero)
            paddings = tf.ones_like(hash_x) * (-1)  # mask成 -1
            hash_x = tf.where(masks, hash_x, paddings)
        #             hash_x = (hash_x + 1) * mask

        return hash_x

    def get_config(self, ):
        config = super(HashLayer, self).get_config()
        config.update({'num_buckets': self.num_buckets, 'mask_zero': self.mask_zero, })
        return config


class Add(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Add, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            return inputs
        if len(inputs) == 1:
            return inputs[0]
        if len(inputs) == 0:
            return tf.constant([[0.0]])
        return tf.keras.layers.add(inputs)


class PredictionLayer(Layer):
    """
      Arguments
         - **task**: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    """

    def __init__(self, task='binary', **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")
        self.task = task
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        # Be sure to call this somewhere!
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.task == "binary":
            x = tf.sigmoid(x)
        output = tf.reshape(x, (-1, 1))

        return output

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'task': self.task}
        base_config = super(PredictionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MMoELayer(Layer):
    """
    The Multi-gate Mixture-of-Experts layer in MMOE model
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - A list with **num_tasks** elements, which is a 2D tensor with shape: ``(batch_size, units_experts)`` .
      Arguments
        - **num_tasks**: integer, the number of tasks, equal to the number of outputs.
        - **num_experts**: integer, the number of experts.
        - **units_experts**: integer, the dimension of each output of MMOELayer.
    References
      - [Jiaqi Ma, Zhe Zhao, Xinyang Yi, et al. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts[C]](https://dl.acm.org/doi/10.1145/3219819.3220007)
    """

    def __init__(self, units_experts, num_experts, num_tasks,
                 use_expert_bias=True, use_gate_bias=True, expert_activation='relu', gate_activation='softmax',
                 expert_bias_initializer='zeros', gate_bias_initializer='zeros', expert_bias_regularizer=None,
                 gate_bias_regularizer=None, expert_bias_constraint=None, gate_bias_constraint=None,
                 expert_kernel_initializer='VarianceScaling', gate_kernel_initializer='VarianceScaling',
                 expert_kernel_regularizer=None, gate_kernel_regularizer=None, expert_kernel_constraint=None,
                 gate_kernel_constraint=None, activity_regularizer=None, **kwargs):
        super(MMoELayer, self).__init__(**kwargs)

        self.units_experts = units_experts
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # Weight parameter
        self.expert_kernels = None
        self.gate_kernels = None
        self.expert_kernel_initializer = tf.keras.initializers.get(expert_kernel_initializer)
        self.gate_kernel_initializer = tf.keras.initializers.get(gate_kernel_initializer)
        self.expert_kernel_regularizer = tf.keras.regularizers.get(expert_kernel_regularizer)
        self.gate_kernel_regularizer = tf.keras.regularizers.get(gate_kernel_regularizer)
        self.expert_kernel_constraint = tf.keras.constraints.get(expert_kernel_constraint)
        self.gate_kernel_constraint = tf.keras.constraints.get(gate_kernel_constraint)

        # Activation parameter
        self.expert_activation = expert_activation
        self.gate_activation = gate_activation

        # Bias parameter
        self.expert_bias = None
        self.gate_bias = None
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.expert_bias_initializer = tf.keras.initializers.get(expert_bias_initializer)
        self.gate_bias_initializer = tf.keras.initializers.get(gate_bias_initializer)
        self.expert_bias_regularizer = tf.keras.regularizers.get(expert_bias_regularizer)
        self.gate_bias_regularizer = tf.keras.regularizers.get(gate_bias_regularizer)
        self.expert_bias_constraint = tf.keras.constraints.get(expert_bias_constraint)
        self.gate_bias_constraint = tf.keras.constraints.get(gate_bias_constraint)

        # Activity parameter
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

        self.expert_layers = []
        self.gate_layers = []

        for i in range(self.num_experts):
            self.expert_layers.append(tf.keras.layers.Dense(self.units_experts, activation=self.expert_activation,
                                                            use_bias=self.use_expert_bias,
                                                            kernel_initializer=self.expert_kernel_initializer,
                                                            bias_initializer=self.expert_bias_initializer,
                                                            kernel_regularizer=self.expert_kernel_regularizer,
                                                            bias_regularizer=self.expert_bias_regularizer,
                                                            activity_regularizer=self.activity_regularizer,
                                                            kernel_constraint=self.expert_kernel_constraint,
                                                            bias_constraint=self.expert_bias_constraint,
                                                            name='expert_net_{}'.format(i)))
        for i in range(self.num_tasks):
            self.gate_layers.append(tf.keras.layers.Dense(self.num_experts, activation=self.gate_activation,
                                                          use_bias=self.use_gate_bias,
                                                          kernel_initializer=self.gate_kernel_initializer,
                                                          bias_initializer=self.gate_bias_initializer,
                                                          kernel_regularizer=self.gate_kernel_regularizer,
                                                          bias_regularizer=self.gate_bias_regularizer,
                                                          activity_regularizer=self.activity_regularizer,
                                                          kernel_constraint=self.gate_kernel_constraint,
                                                          bias_constraint=self.gate_bias_constraint,
                                                          name='gate_net_{}'.format(i)))

    def call(self, inputs, **kwargs):

        expert_outputs, gate_outputs, final_outputs = [], [], []

        # inputs: (batch_size, embedding_size)
        for expert_layer in self.expert_layers:
            expert_output = tf.expand_dims(expert_layer(inputs), axis=2)
            expert_outputs.append(expert_output)

        # batch_size * units * num_experts
        expert_outputs = tf.concat(expert_outputs, 2)

        # [(batch_size, num_experts), ......]
        for gate_layer in self.gate_layers:
            gate_outputs.append(gate_layer(inputs))

        for gate_output in gate_outputs:
            # (batch_size, 1, num_experts)
            expanded_gate_output = tf.expand_dims(gate_output, axis=1)

            # (batch_size * units * num_experts) * (batch_size, 1 * units, num_experts)
            weighted_expert_output = expert_outputs * tf.keras.backend.repeat_elements(expanded_gate_output,
                                                                                       self.units_experts, axis=1)

            # (batch_size, units)
            final_outputs.append(tf.reduce_sum(weighted_expert_output, axis=2))

        # [(batch_size, units), ......]   size: num_task
        return final_outputs

    def get_config(self, ):
        config = {'units_experts': self.units_experts, 'num_experts': self.num_experts, 'num_tasks': self.num_tasks}
        base_config = super(MMoELayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DNN(Layer):
    """The Multi Layer Percetron
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.
      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function to use.
        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, hidden_units, activation='relu', use_bias=True, bias_initializer='zeros',
                 bias_regularizer=None, bias_constraint=None, kernel_initializer='VarianceScaling',
                 kernel_regularizer=None, kernel_constraint=None,
                 activity_regularizer=None, seed=1024, **kwargs):

        # Weight parameter
        self.kernels = None
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)

        # Activation parameter
        self.activation = activation

        # Bias parameter
        self.bias = None
        self.use_bias = use_bias
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        # Activity parameter
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

        # hidden_units parameter
        self.hidden_units = hidden_units
        self.seed = seed

        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dnn_layer = [Dense(units=self.hidden_units[i],
                                activation=self.activation,
                                use_bias=self.use_bias,
                                kernel_initializer=self.kernel_initializer,
                                bias_initializer=self.bias_initializer,
                                kernel_regularizer=self.kernel_regularizer,
                                bias_regularizer=self.bias_regularizer,
                                activity_regularizer=self.activity_regularizer,
                                kernel_constraint=self.kernel_constraint,
                                bias_constraint=self.bias_constraint,
                                ) for i in range(len(self.hidden_units))]

        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        dnn_input = inputs

        # DNN part
        for i in range(len(self.hidden_units)):
            if i == len(self.hidden_units) - 1:
                dnn_out = self.dnn_layer[i](dnn_input)
                break

            dnn_input = self.dnn_layer[i](dnn_input)

        return dnn_out

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units, 'seed': self.seed,
                  'use_bias': self.use_bias, 'kernel_initializer': self.kernel_initializer,
                  'bias_initializer': self.bias_initializer, 'kernel_regularizer': self.kernel_regularizer,
                  'bias_regularizer': self.bias_regularizer, 'activity_regularizer': self.activity_regularizer,
                  'kernel_constraint': self.kernel_constraint, 'bias_constraint': self.bias_constraint, }
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


########################################################################
               #################定义输入帮助函数##############
########################################################################

# 定义model输入特征
def build_input_features(features_columns, prefix=''):
    input_features = OrderedDict()

    for feat_col in features_columns:
        if isinstance(feat_col, DenseFeat):
            input_features[feat_col.name] = Input([feat_col.dim], name=feat_col.name)
        elif isinstance(feat_col, SparseFeat):
            input_features[feat_col.name] = Input([1], name=feat_col.name, dtype=feat_col.dtype)
        elif isinstance(feat_col, VarLenSparseFeat):
            input_features[feat_col.name] = Input([None], name=feat_col.name, dtype=feat_col.dtype)
            if feat_col.weight_name is not None:
                input_features[feat_col.weight_name] = Input([None], name=feat_col.weight_name, dtype='float32')
        else:
            raise TypeError("Invalid feature column in build_input_features: {}".format(feat_col.name))

    return input_features


# 构造 自定义embedding层 matrix
def build_embedding_matrix(features_columns, linear_dim=None):
    embedding_matrix = {}
    for feat_col in features_columns:
        if isinstance(feat_col, SparseFeat) or isinstance(feat_col, VarLenSparseFeat):
            vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
            vocab_size = feat_col.voc_size + 2
            embed_dim = feat_col.embed_dim if linear_dim is None else 1
            name_tag = '' if linear_dim is None else '_linear'
            if vocab_name not in embedding_matrix:
                embedding_matrix[vocab_name] = tf.Variable(
                    initial_value=tf.random.truncated_normal(shape=(vocab_size, embed_dim), mean=0.0,
                                                             stddev=0.001, dtype=tf.float32), trainable=True,
                    name=vocab_name + '_embed' + name_tag)
    return embedding_matrix

# 构造 自定义embedding层
def build_embedding_dict(features_columns):
    embedding_dict = {}
    embedding_matrix = build_embedding_matrix(features_columns)

    for feat_col in features_columns:
        if isinstance(feat_col, SparseFeat):
            vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
            embedding_dict[feat_col.name] = EmbeddingLookup(embedding=embedding_matrix[vocab_name],
                                                            name='emb_lookup_' + feat_col.name)
        elif isinstance(feat_col, VarLenSparseFeat):
            vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
            if feat_col.combiner is not None:
                if feat_col.weight_name is not None:
                    embedding_dict[feat_col.name] = EmbeddingLookupSparse(embedding=embedding_matrix[vocab_name],
                                                                          combiner=feat_col.combiner, has_weight=True,
                                                                          name='emb_lookup_sparse_' + feat_col.name)
                else:
                    embedding_dict[feat_col.name] = EmbeddingLookupSparse(embedding=embedding_matrix[vocab_name],
                                                                          combiner=feat_col.combiner,
                                                                          name='emb_lookup_sparse_' + feat_col.name)
            else:
                embedding_dict[feat_col.name] = EmbeddingLookup(embedding=embedding_matrix[vocab_name],
                                                                name='emb_lookup_' + feat_col.name)

    return embedding_dict


# 构造 自定义embedding层
def build_linear_embedding_dict(features_columns):
    embedding_dict = {}
    embedding_matrix = build_embedding_matrix(features_columns, linear_dim=1)
    name_tag = '_linear'

    for feat_col in features_columns:
        if isinstance(feat_col, SparseFeat):
            vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
            embedding_dict[feat_col.name] = EmbeddingLookup(embedding=embedding_matrix[vocab_name],
                                                            name='emb_lookup_' + feat_col.name + name_tag)
        elif isinstance(feat_col, VarLenSparseFeat):
            vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
            if feat_col.combiner is not None:
                if feat_col.weight_name is not None:
                    embedding_dict[feat_col.name] = EmbeddingLookupSparse(embedding=embedding_matrix[vocab_name],
                                                                          combiner=feat_col.combiner, has_weight=True,
                                                                          name='emb_lookup_sparse_' + feat_col.name + name_tag)
                else:
                    embedding_dict[feat_col.name] = EmbeddingLookupSparse(embedding=embedding_matrix[vocab_name],
                                                                          combiner=feat_col.combiner,
                                                                          name='emb_lookup_sparse_' + feat_col.name + name_tag)
            else:
                embedding_dict[feat_col.name] = EmbeddingLookup(embedding=embedding_matrix[vocab_name],
                                                                name='emb_lookup_' + feat_col.name + name_tag)

    return embedding_dict


# dense 与 embedding特征输入
def input_from_feature_columns(features, features_columns, embedding_dict, cate_map=CATEGORICAL_MAP):
    sparse_embedding_list = []
    dense_value_list = []

    for feat_col in features_columns:
        if isinstance(feat_col, SparseFeat):
            _input = features[feat_col.name]
            if feat_col.vocab is not None:
                vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
                keys = cate_map[vocab_name]
                _input = VocabLayer(keys)(_input)
            elif feat_col.hash_size is not None:
                _input = HashLayer(num_buckets=feat_col.hash_size, mask_zero=False)(_input)

            embed = embedding_dict[feat_col.name](_input)
            sparse_embedding_list.append(embed)
        elif isinstance(feat_col, VarLenSparseFeat):
            _input = features[feat_col.name]
            if feat_col.vocab is not None:
                mask_val = '-1' if feat_col.dtype == 'string' else -1
                vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
                keys = cate_map[vocab_name]
                _input = VocabLayer(keys, mask_value=mask_val)(_input)
            elif feat_col.hash_size is not None:
                _input = HashLayer(num_buckets=feat_col.hash_size, mask_zero=True)(_input)
            if feat_col.combiner is not None:
                input_sparse = DenseToSparseTensor(mask_value=-1)(_input)
                if feat_col.weight_name is not None:
                    weight_sparse = DenseToSparseTensor()(features[feat_col.weight_name])
                    embed = embedding_dict[feat_col.name]([input_sparse, weight_sparse])
                else:
                    embed = embedding_dict[feat_col.name](input_sparse)
            else:
                embed = embedding_dict[feat_col.name](_input)

            sparse_embedding_list.append(embed)

        elif isinstance(feat_col, DenseFeat):
            dense_value_list.append(features[feat_col.name])

        else:
            raise TypeError("Invalid feature column in input_from_feature_columns: {}".format(feat_col.name))

    return sparse_embedding_list, dense_value_list


def concat_func(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return Concatenate(axis=axis)(inputs)


def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = Flatten()(concat_func(sparse_embedding_list))
        dense_dnn_input = Flatten()(concat_func(dense_value_list))
        return concat_func([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return Flatten()(concat_func(sparse_embedding_list))
    elif len(dense_value_list) > 0:
        return Flatten()(concat_func(dense_value_list))
    else:
        raise "dnn_feature_columns can not be empty list"


def get_linear_logit(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_linear_layer = Add()(sparse_embedding_list)
        sparse_linear_layer = Flatten()(sparse_linear_layer)
        dense_linear = concat_func(dense_value_list)
        dense_linear_layer = Dense(1)(dense_linear)
        linear_logit = Add(name='linear_logit')([dense_linear_layer, sparse_linear_layer])
        return linear_logit
    elif len(sparse_embedding_list) > 0:
        sparse_linear_layer = Add()(sparse_embedding_list)
        sparse_linear_layer = Flatten(name='linear_logit')(sparse_linear_layer)
        return sparse_linear_layer
    elif len(dense_value_list) > 0:
        dense_linear = concat_func(dense_value_list)
        dense_linear_layer = Dense(1, name='linear_logit')(dense_linear)
        return dense_linear_layer
    else:
        raise "linear_feature_columns can not be empty list"


########################################################################
               #################定义模型##############
########################################################################

def MMOE(dnn_feature_columns, num_tasks, tasks, tasks_name, num_experts=4, units_experts=128, task_dnn_units=(32, 32),
         seed=1024, dnn_activation='relu'):
    """Instantiates the Multi-gate Mixture-of-Experts architecture.
    Args:
        dnn_feature_columns: An iterable containing all the features used by deep part of the model.
        num_tasks: integer, number of tasks, equal to number of outputs, must be greater than 1.
        tasks: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
        num_experts: integer, number of experts.
        units_experts: integer, the hidden units of each expert.
        tasks_name: list of str, the name of each tasks,
        task_dnn_units: list,list of positive integer or empty list, the layer number and units in each layer of task-specific DNN
        dnn_activation: Activation function to use in DNN
    return: return a Keras model instance.
    """
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(tasks) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of tasks")
    for task in tasks:
        if task not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task))
    # 特征输入
    features = build_input_features(dnn_feature_columns)  # {n1: Input, n2:Input}

    inputs_list = list(features.values())

    # 构建 dnn embedding_dict
    dnn_embedding_dict = build_embedding_dict(dnn_feature_columns)
    dnn_sparse_embedding_list, dnn_dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                                 dnn_embedding_dict)
    dnn_input = combined_dnn_input(dnn_sparse_embedding_list, dnn_dense_value_list)

    # MMOELayer
    mmoe_layers = MMoELayer(units_experts=units_experts, num_tasks=num_tasks, num_experts=num_experts,
                            name='mmoe_layer')(dnn_input)

    # 分别处理不同 Task Tower
    task_outputs = []
    for task_layer, task, task_name in zip(mmoe_layers, tasks, tasks_name):
        tower_layer = DNN(hidden_units=task_dnn_units, activation='relu', name='tower_{}'.format(task_name))(task_layer)

        # batch_size * 1
        output_layer = tf.keras.layers.Dense(units=1, activation=None, use_bias=False,
                                             kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                             name='logit_{}'.format(task_name))(tower_layer)
        output = PredictionLayer(task, name=task_name)(output_layer)

        task_outputs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=task_outputs)

    return model


########################################################################
               #################模型训练##############
########################################################################

model = MMOE(
    dnn_feature_columns=dnn_feature_columns,
    num_tasks=2,
    tasks=['binary', 'regression'],
    tasks_name=['CTR', 'DUR'])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.7, beta_2=0.8),
              loss={"CTR": "binary_crossentropy",
                    "DUR": "mse",
                    },
              loss_weights=[1.0, 1.0],
              metrics={"CTR": [tf.keras.metrics.AUC(name='auc')],
                       "DUR": ["mae"]}
              )

log_dir = './mywork/tensorboardshare/logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tbCallBack = TensorBoard(log_dir=log_dir,  # log 目录
                         histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                         write_graph=True,  # 是否存储网络结构图
                         write_images=True,  # 是否可视化参数
                         update_freq='epoch',
                         embeddings_freq=0,
                         embeddings_layer_names=None,
                         embeddings_metadata=None,
                         profile_batch='2,2')

total_train_sample = 500
total_test_sample = 50
train_steps_per_epoch = np.floor(total_train_sample / batch_size).astype(np.int32)
test_steps_per_epoch = np.ceil(total_test_sample / val_batch_size).astype(np.int32)
history_loss = model.fit(dataset, epochs=1,
                         steps_per_epoch=train_steps_per_epoch,
                         validation_data=dataset_val, validation_steps=test_steps_per_epoch,
                         verbose=1, callbacks=[tbCallBack])
