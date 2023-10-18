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
import tensorflow as tf
#import tensorflow_recommenders_addons as tfra
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config


class DynamicEmbeddingV2(tf.keras.layers.Layer):
    def __init__(self,
                 embedding_dim,
                 key_dtype=tf.int64,
                 embeddings_initializer='uniform',
                 **kwargs):
        super(DynamicEmbeddingV2, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.key_dtype = key_dtype
        self.embeddings_initializer = tf.keras.initializers.get(embeddings_initializer)

    def build(self, input_shape):
        if context.executing_eagerly() and tf_config.list_logical_devices('GPU'):
            with tf.device('cpu:0'):
                self.embeddings = tfra.dynamic_embedding.get_variable(
                    name="dynamic_embeddings_v2",
                    dim=self.embedding_dim,
                    key_dtype=self.key_dtype,
                    initializer=self.embeddings_initializer)
        else:
            self.embeddings = tfra.dynamic_embedding.get_variable(
                name="dynamic_embeddings_v2",
                dim=self.embedding_dim,
                key_dtype=self.key_dtype,
                initializer=self.embeddings_initializer)
        self.built = True

    def call(self, inputs, **kwargs):
        out = tfra.dynamic_embedding.embedding_lookup_unique(params=self.embeddings, ids=inputs)
        return out


class DynamicEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 embedding_dim,
                 key_dtype=tf.int64,
                 embeddings_initializer='uniform',
                 **kwargs):
        super(DynamicEmbedding, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.key_dtype = key_dtype
        self.embeddings_initializer = tf.keras.initializers.get(embeddings_initializer)

    def build(self, input_shape):
        if context.executing_eagerly() and tf_config.list_logical_devices('GPU'):
            with tf.device('cpu:0'):
                self.embeddings = tfra.embedding_variable.EmbeddingVariable(
                    name="dynamic_embeddings",
                    embedding_dim=self.embedding_dim,
                    ktype=self.key_dtype,
                    initializer=self.embeddings_initializer)
        else:
            self.embeddings = tfra.embedding_variable.EmbeddingVariable(
                name="dynamic_embeddings",
                embedding_dim=self.embedding_dim,
                ktype=self.key_dtype,
                initializer=self.embeddings_initializer)
        self.built = True

    def call(self, inputs, **kwargs):
        val, idx = tf.unique(inputs)
        weights = tf.nn.embedding_lookup(
            params=self.embeddings,
            ids=val,
            name="dynamic_lookup")
        out = tf.gather(weights, idx)
        return out


class Embedding(tf.keras.layers.Embedding):
    def __init__(self,
                 input_dim,
                 output_dim,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 **kwargs):
        super(Embedding, self).__init__(input_dim=input_dim,
                                        output_dim=output_dim,
                                        embeddings_initializer=embeddings_initializer,
                                        embeddings_regularizer=embeddings_regularizer,
                                        activity_regularizer=activity_regularizer,
                                        embeddings_constraint=embeddings_constraint,
                                        mask_zero=mask_zero,
                                        input_length=input_length,
                                        **kwargs)

    def call(self, inputs):
        dtype = tf.keras.backend.dtype(inputs)
        if dtype != 'int32' and dtype != 'int64':
            inputs = tf.cast(inputs, 'int32')
        shape = tf.shape(inputs)
        ids_flat = tf.reshape(inputs, tf.reduce_prod(shape, keepdims=True))
        unique_ids, idx = tf.unique(ids_flat)
        unique_embeddings = tf.nn.embedding_lookup(self.embeddings, unique_ids)
        embeddings_flat = tf.gather(unique_embeddings, idx)
        embeddings_shape = tf.concat(
            [shape, tf.shape(unique_embeddings)[1:]], 0)
        out = tf.reshape(embeddings_flat, embeddings_shape)
        out.set_shape(inputs.get_shape().concatenate(
            unique_embeddings.get_shape()[1:]))
        if self._dtype_policy.compute_dtype != self._dtype_policy.variable_dtype:
            # Instead of casting the variable as in most layers, cast the output, as
            # this is mathematically equivalent but is faster.
            out = tf.cast(out, self._dtype_policy.compute_dtype)
        return out


class EmbeddingMBA(tf.keras.layers.Embedding):
    """ Embedding添加频次L2正则化

    Args:
        l2: l2正则化系数
        freqs: 特征在样本中出现的频次，从0开始编号，例如：[5, 10, 23]，表示id分别=[0,1,2]的特征频次
        freq_smooth: 频次平滑系数，默认值0.5
        total_freqs: 总词频，如果指定总词频，则不再根据freqs计算总词频
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 l2=0.0,
                 freqs=None,
                 freq_smooth=0.5,
                 total_freqs=None,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 **kwargs):
        self.freqs = freqs
        self.l2 = l2
        self.freq_smooth = freq_smooth
        self.total_freqs = total_freqs
        super(EmbeddingMBA, self).__init__(input_dim=input_dim,
                                           output_dim=output_dim,
                                           embeddings_initializer=embeddings_initializer,
                                           embeddings_regularizer=embeddings_regularizer,
                                           activity_regularizer=activity_regularizer,
                                           embeddings_constraint=embeddings_constraint,
                                           mask_zero=mask_zero,
                                           input_length=input_length,
                                           **kwargs)

    def build(self, input_shape):
        # TODO(rabin) 效果会存在万2的差异，未明确原因
        """coef = np.array(self.freqs)
        if self.total_freqs is None:
            # coef.sum(axis=0) = 所有特征的词频加和
            self.total_freqs = coef.sum()
        coef = self.total_freqs / (coef + self.freq_smooth)
        coef = np.array(coef).reshape([-1, 1])
        coef = self.add_weight(
            shape=(len(self.freqs), 1),
            initializer=tf.keras.initializers.constant(coef),
            name='freqs_coef',
            trainable=False
        )"""
        coef = [[i for j in range(self.output_dim)] for i in self.freqs]
        coef = np.array(coef)
        if self.total_freqs is None:
            self.total_freqs = coef.sum(axis=0)
        coef = self.total_freqs / (coef + self.freq_smooth)
        coef = self.add_weight(
            shape=(len(self.freqs), self.output_dim),
            initializer=tf.keras.initializers.constant(coef),
            name='freqs_coef',
            trainable=False
        )

        super(EmbeddingMBA, self).build(input_shape)
        self.add_loss(tf.math.reduce_sum(self.l2 * tf.math.square(self.embeddings) * coef))

    def get_config(self):
        config = {
            'freqs': self.freqs,
            'l2': self.l2,
            'freq_smooth': self.freq_smooth,
            'total_freqs': self.total_freqs
        }
        base_config = super(EmbeddingMBA, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

