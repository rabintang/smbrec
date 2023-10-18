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
import itertools

import tensorflow as tf
import smbrec.layers
from tensorflow.python.keras import backend as K


class FieldWiseBiInteraction(tf.keras.layers.Layer):
    """ 基于attention的序列汇聚

    call的输入有三个， query, keys, keys_mask，其中：
    query 是 batch_size * embedding_size 维度向量，
    keys 是 batch_size * sequence_len * embedding_size
    keys_mask 是 batch_size * sequence_len

    Args:
        hidden_units: 全连接的隐层单元数
        activation: 全连接的激活函数
        dropout: 全连接的dropout
        kernel_initializer: 全连接的kernel初始化器
        **kwargs:
    """

    def __init__(self, use_bias=True, **kwargs):
        self.use_bias = use_bias
        super(FieldWiseBiInteraction, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                'A `Field-Wise Bi-Interaction` layer should be called '
                'on a tenor of 3dims')

        self.num_fields = input_shape[1]
        embedding_size = input_shape[-1]

        self.kernel_mf = self.add_weight(
            name='kernel_mf',
            shape=(int(self.num_fields * (self.num_fields - 1) / 2), 1),
            initializer=tf.keras.initializers.Ones(),
            regularizer=None,
            trainable=True)

        if self.use_bias:
            self.bias_mf = self.add_weight(name='bias_mf',
                                           shape=(embedding_size),
                                           initializer=tf.initializers.Zeros())

        super(FieldWiseBiInteraction, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" %
                (K.ndim(inputs)))

        field_wise_vectors = inputs

        # MF module
        left = []
        right = []

        for i, j in itertools.combinations(list(range(self.num_fields)), 2):
            left.append(i)
            right.append(j)

        embeddings_left = tf.gather(params=field_wise_vectors,
                                    indices=left,
                                    axis=1)
        embeddings_right = tf.gather(params=field_wise_vectors,
                                     indices=right,
                                     axis=1)

        embeddings_prod = embeddings_left * embeddings_right
        field_weighted_embedding = embeddings_prod * self.kernel_mf
        h_mf = tf.reduce_sum(field_weighted_embedding, axis=1)
        if self.use_bias:
            h_mf = tf.nn.bias_add(h_mf, self.bias_mf)

        return h_mf

    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][-1])

    def get_config(self):
        config = {
            'use_bias': self.use_bias
        }
        base_config = super(FieldWiseBiInteraction, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
