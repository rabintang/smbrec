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


class DNN(tf.keras.layers.Layer):
    """ 多层全连接网络
    每层全连接都可以接dropout和batch norm

    Args:
        units: 隐层单元大小数组
        activation: 激活函数
        dropout: dropout rate
        use_bn: 是否添加batch norm
        kernel_initializer: kernel初始化器
        bias_initializer: bias初始化器
        kernel_regularizer: kernel正则化器
        bias_regularizer: bias正则化器
        **kwargs:
    """

    def __init__(self,
                 units,
                 activation='relu',
                 dropout=0,
                 use_bn=False,
                 output_activation='SAME',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        self.units = units
        self.activation = activation
        self.dropout = dropout
        self.use_bn = use_bn
        self.output_activation = activation if output_activation == 'SAME' else output_activation
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.use_bn:
            self.bns = [tf.keras.layers.BatchNormalization() for _ in range(len(self.units))]
        self.dropouts = [tf.keras.layers.Dropout(self.dropout) for _ in range(len(self.units))]
        if tf.__version__ < '1.15.0':
            self.activations = [tf.keras.layers.Activation(self.activation) for _ in range(len(self.units))]
        else:
            self.activations = [tf.keras.activations.get(self.activation) for _ in range(len(self.units))]

        if self.output_activation:
            if tf.__version__ < '1.15.0':
                self.activations[-1] = tf.keras.layers.Activation(self.output_activation)
            else:
                self.activations[-1] = tf.keras.activations.get(self.output_activation)

        if tf.__version__ < '1.15.0':
            self.nn = tf.keras.Sequential()
            for idx, hidden_unit in enumerate(self.units):
                self.nn.add(tf.keras.layers.Dense(hidden_unit,
                                                  kernel_initializer=self.kernel_initializer,
                                                  kernel_regularizer=self.kernel_regularizer,
                                                  bias_initializer=self.bias_initializer,
                                                  bias_regularizer=self.bias_regularizer))
                if self.use_bn:
                    self.nn.add(self.bns[idx])
                self.nn.add(self.activations[idx])
                self.nn.add(self.dropouts[idx])
        else:
            input_size = input_shape[-1]
            hidden_units = [int(input_size)] + list(self.units)
            self.kernels = [self.add_weight(name='kernel' + str(i),
                                            shape=(hidden_units[i], hidden_units[i + 1]),
                                            initializer=self.kernel_initializer,
                                            regularizer=self.kernel_regularizer,
                                            trainable=True) for i in range(len(self.units))]
            self.biases = [self.add_weight(name='bias' + str(i),
                                           shape=(self.units[i],),
                                           initializer=self.bias_initializer,
                                           regularizer=self.bias_regularizer,
                                           trainable=True) for i in range(len(self.units))]

        super(DNN, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        if tf.__version__ < '1.15.0':
            deep_input = self.nn(inputs, training=training)
        else:
            deep_input = inputs
            for i in range(len(self.units)):
                fc = tf.nn.bias_add(tf.tensordot(
                    deep_input, self.kernels[i], axes=(-1, 0)), self.biases[i])
                if self.use_bn:
                    fc = self.bns[i](fc, training=training)
                try:
                    fc = self.activations[i](fc, training=training)
                except TypeError as e:  # TypeError: call() got an unexpected keyword argument 'training'
                    print("make sure the activation function use training flag properly", e)
                    fc = self.activations[i](fc)
                fc = self.dropouts[i](fc, training=training)
                deep_input = fc

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.units) > 0:
            shape = input_shape[:-1] + (self.units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self):
        config = {'activation': self.activation,
                  'units': self.units,
                  'use_bn': self.use_bn,
                  'dropout': self.dropout,
                  'output_activation': self.output_activation,
                  'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
                  'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
                  'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer)
                  }
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
