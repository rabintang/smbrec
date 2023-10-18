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

import json
import tensorflow as tf
from smbrec.utils import file_utils


class VocabEncoder(tf.keras.layers.Layer):
    """ 词表编码
    基于给定的词表，对输入的数据进行编码

    Args:
        keys: 词表，可以是 list 或者 词表文件的地址，支持hdfs文件
        default: OOV的默认值，默认0
        begin_index: 编码的起始编码，默认0
        key_dtype: key的数据类型，采用字符串名，tf.dtype 无法直接序列化
        **kwargs: Layer的其他参数
    """

    def __init__(self, keys, default=0, begin_index=0, key_dtype=None, **kwargs):
        super(VocabEncoder, self).__init__(**kwargs)
        self.keys = keys
        self.default = default
        self.begin_index = begin_index
        self.key_dtype = key_dtype
        key_dtype = key_dtype if key_dtype is None else getattr(tf, key_dtype)
        # 如果是文件，则从文件中读取
        if isinstance(keys, str):
            keys = file_utils.readall(keys).split('\n')
            if key_dtype in (tf.int8, tf.int16, tf.int32, tf.int64):
                keys = [int(key) for key in keys]
            elif key_dtype in (tf.float32, tf.double):
                keys = [float(key) for key in keys]
        vals = list(range(begin_index, begin_index + len(keys)))
        vals = tf.constant(vals, dtype=tf.int64)
        keys = tf.constant(keys, dtype=key_dtype)
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, vals), default)

    def call(self, inputs, **kwargs):
        idx = self.table.lookup(inputs)
        return idx

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(VocabEncoder, self).get_config()
        config.update({
            'keys': self.keys,
            'default': self.default,
            'begin_index': self.begin_index,
            'key_dtype': self.key_dtype
        })
        return config


class GroupVocabEncoder(tf.keras.layers.Layer):
    """ 特征组词表编码
    基于给定的词表，对输入的数据进行编码

    Args:
        keys: 词表，可以是 dict 或者 词表文件的地址，支持hdfs文件，词表为json格式，如：{column1: [v1,v2,v3], column2: [v1,v2]}
        columns: 特征列名，list，需要保证与inputs的第二维顺序相同
        default: OOV的默认值，默认0
        begin_index: 编码的起始编码，默认0
        key_dtype: key的数据类型，采用字符串名，tf.dtype 无法直接序列化
        **kwargs: Layer的其他参数
    """

    def __init__(self, keys, columns, default=0, begin_index=0, key_dtype=None, **kwargs):
        super(GroupVocabEncoder, self).__init__(**kwargs)
        self.keys = keys
        self.columns = columns
        self.default = default
        self.begin_index = begin_index
        self.key_dtype = key_dtype
        key_dtype = key_dtype if key_dtype is None else getattr(tf, key_dtype)
        self.tables = {}
        # 如果是文件，则从文件中读取
        if isinstance(keys, str):
            keys = json.loads(file_utils.readall(keys))
        for column in self.columns:
            key_val = keys[column]
            if key_dtype in (tf.int8, tf.int16, tf.int32, tf.int64):
                key_val = [int(v) for v in keys[column]]
            elif key_dtype in (tf.float32, tf.double):
                key_val = [float(v) for v in keys[column]]
            vals = list(range(begin_index, begin_index + len(key_val)))
            vals = tf.constant(vals, dtype=tf.int64)
            key_val = tf.constant(key_val, dtype=key_dtype)
            self.tables[column] = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(key_val, vals), default)

    def call(self, inputs, **kwargs):
        input_shape = inputs.get_shape()
        assert len(input_shape) == 2 or len(input_shape) == 3,\
            "input shape is %d, only support 2 or 3." % len(input_shape)
        output = []
        for idx, column in enumerate(self.columns):
            if len(input_shape) == 3:
                output.append(self.tables[column].lookup(inputs[:, idx, :]))
            else:
                output.append(tf.expand_dims(self.tables[column].lookup(inputs[:, idx]), axis=1))

        return tf.concat(output, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(GroupVocabEncoder, self).get_config()
        config.update({
            'keys': self.keys,
            'columns': self.columns,
            'default': self.default,
            'begin_index': self.begin_index,
            'key_dtype': self.key_dtype
        })
        return config
