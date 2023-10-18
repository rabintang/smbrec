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

import tensorflow as tf
import smbrec.layers


class PositionEncoding(tf.keras.layers.Layer):
    """ 序列位置偏置编码

    
    Args:
        pos_embedding_initializer: 位置embedding参数初始化方式，sin_cos表示正余弦，其他初始化方式同tf
        pos_embedding_trainable: 是否训练参数，默认True
        max_len: 序列最大长度，默认为None
        zero_pad: 序列第一个位置embedding为0
        scale: 是否缩放
    """

    def __init__(self,
                 pos_embedding_initializer='sin_cos',
                 pos_embedding_trainable=True,
                 max_len=None,
                 zero_pad=False,
                 scale=True,
                 **kwargs):
        self.pos_embedding_initializer = pos_embedding_initializer
        self.pos_embedding_trainable = pos_embedding_trainable
        self.max_len = max_len
        self.zero_pad = zero_pad
        self.scale = scale
        super(PositionEncoding, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        _, T, num_units = input_shape.as_list()  # inputs.get_shape().as_list()
        max_len = self.max_len if self.max_len is not None else T

        if self.pos_embedding_initializer == "sin_cos":
            # First part of the PE function: sin and cos argument
            position_enc = np.array([
                [pos / np.power(10000, 2. * (i // 2) / num_units) for i in range(num_units)]
                for pos in range(max_len)])
    
            # Second part, apply the cosine to even columns and sin to odds.
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
            if self.zero_pad:
                position_enc[0, :] = np.zeros(num_units)
            initializer = Constant(position_enc)
        else:
            # TODO(rabin): 其他初始化方式暂时不支持zero_pad
            initializer = self.pos_embedding_initializer
        self.lookup_table = self.add_weight("lookup_table", (max_len, num_units),
                                            initializer=initializer,
                                            trainable=self.pos_embedding_trainable)

        # Be sure to call this somewhere!
        super(PositionEncoding, self).build(input_shape)

    def call(self, inputs, mask=None):
        _, T, num_units = inputs.get_shape().as_list()
        position_ind = tf.expand_dims(tf.range(T), 0)
        outputs = tf.nn.embedding_lookup(self.lookup_table, position_ind)
        if self.scale:
            outputs = outputs * num_units ** 0.5
        return outputs + inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = {'pos_embedding_trainable': self.pos_embedding_trainable,
                  'pos_embedding_initializer': "sin_cons" if self.pos_embedding_initializer == "sin_cons" \
                      else tf.keras.initializers.serialize(self.pos_embedding_initializer),
                  'zero_pad': self.zero_pad,
                  'max_len': self.max_len,
                  'scale': self.scale}
        base_config = super(PositionEncoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class AttentionPooling(tf.keras.layers.Layer):
    """ 基于attention的序列汇聚

    call的输入有三个， query, keys, keys_mask，其中：
    query 是 batch_size * embedding_size 维度向量，
    keys 是 batch_size * sequence_len * embedding_size
    keys_mask 是 batch_size * sequence_len

    Args:
        hidden_units: 全连接的隐层单元数
        activation: 全连接的激活函数
        att_activation: 最后一层attention激活函数
        dropout: 全连接的dropout
        kernel_initializer: 全连接的kernel初始化器
        **kwargs:
    """

    def __init__(self,
                 att_units=(80, 40),
                 att_activation='relu',
                 att_output_activation='sigmoid',
                 att_dropout=0.1,
                 add_position_encoding=True,
                 max_len=None,
                 kernel_initializer='glorot_normal',
                 pos_embedding_initializer='glorot_normal',
                 **kwargs):
        self.att_units = att_units
        self.att_dropout = att_dropout
        self.att_activation = att_activation
        self.att_output_activation = att_output_activation
        self.add_position_encoding = add_position_encoding
        self.max_len = max_len
        self.pos_embedding_initializer = 'sin_cos' if pos_embedding_initializer == 'sin_cos' \
                else tf.keras.initializers.get(pos_embedding_initializer)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        super(AttentionPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3, 'A `AttentionPooling` layer should be called on a list of 3 inputs'
        if len(input_shape[0]) != 2 or len(input_shape[1]) != 3 or len(input_shape[2]) != 2:
            raise ValueError(
                "Unexpected inputs dimensions,the 3 tensor dimensions are %d,%d and %d , expect to be 2,3 and 2" % (
                    len(input_shape[0]), len(input_shape[1]), len(input_shape[2])))
        if input_shape[0][-1] != input_shape[1][-1] or input_shape[1][1] != input_shape[2][1]:
            raise ValueError('A `AttentionPooling` layer requires inputs of a 3 tensor with shape '
                             '(None,embedding_size),(None,T,embedding_size) and (None,T)'
                             'Got different shapes: %s' % input_shape)

        self.nn = smbrec.layers.DNN(self.att_units + [1],
                                    kernel_initializer=self.kernel_initializer,
                                    activation=self.att_activation,
                                    output_activation=self.att_output_activation,
                                    dropout=self.att_dropout)
        if self.add_position_encoding:
            self.query_pos_encoding = smbrec.layers.PositionEncoding(
                    max_len=self.max_len,
                    scale=False,
                    pos_embedding_initializer=self.pos_embedding_initializer)
            self.key_pos_encoding = smbrec.layers.PositionEncoding(
                    max_len=self.max_len,
                    scale=False,
                    pos_embedding_initializer=self.pos_embedding_initializer)
        super(AttentionPooling, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        queries, keys, key_masks = inputs  # [batch, dim] [batch, seq, dim], [batch, seq]
        # 添加position embedding
        if self.add_position_encoding:
            keys = self.key_pos_encoding(keys)
            queries = tf.squeeze(self.query_pos_encoding(tf.expand_dims(queries, axis=1)), axis=1)
        key_masks = tf.expand_dims(tf.cast(key_masks, tf.float32), 1)  # [batch, 1, seq]
        queries = tf.tile(tf.expand_dims(queries, 1), (1, tf.shape(keys)[1], 1))
        att_input = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
        att_score = self.nn(att_input, training=training)  # [batch, seq, 1]
        att_score = tf.transpose(att_score, (0, 2, 1))  # [batch, 1, seq]
        outputs = tf.squeeze(tf.matmul(key_masks * att_score, keys), axis=1)  # [batch, dim]
        return outputs

    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][-1])

    def get_config(self):
        config = {
            'att_units': self.att_units,
            'att_activation': self.att_activation,
            'att_output_activation': self.att_output_activation,
            'att_dropout': self.att_dropout,
            'add_position_encoding': self.add_position_encoding,
            'max_len': self.max_len,
            'pos_embedding_initializer': 'sin_cos' if self.pos_embedding_initializer == 'sin_cos' \
                    else tf.keras.initializers.serialize(self.pos_embedding_initializer),
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer)
        }
        base_config = super(AttentionPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
