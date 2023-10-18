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


class SequenceSlice(tf.keras.layers.Layer):
    """ 序列截断并转Dense Tensor

    Args:
        max_len: 序列截断的最大长度
        pad: 是否补足最大长度，默认False
    """

    def __init__(self, max_len, pad=False, **kwargs):
        self.max_len = max_len
        self.pad = pad
        super(SequenceSlice, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        shape = tf.shape(inputs)
        new_shape = []
        for i in range(len(inputs.get_shape()) - 1):
            new_shape.append(shape[i])
        new_shape.append(self.max_len)
        slice = tf.sparse.slice(inputs, [0] * len(inputs.get_shape()), new_shape)
        dense_slice = tf.sparse.to_dense(slice)
        if self.pad:
            paddings = [[0, 0], [0, self.max_len - tf.shape(dense_slice)[1]]]
            dense_slice = tf.reshape(tf.pad(dense_slice, paddings, 'CONSTANT'), (-1, self.max_len))
        return dense_slice

    def get_config(self):
        config = {
            'max_len': self.max_len,
            'pad': self.pad
        }
        base_config = super(SequenceSlice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
