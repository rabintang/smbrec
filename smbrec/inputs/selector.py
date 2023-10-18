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

import gzip
import tensorflow as tf
from tensorflow.python.data.experimental.ops.readers import _infer_column_names
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io

from smbrec.core.config import SampleFormat
from smbrec.inputs.tfrecord_input import TFRecordInput
from smbrec.inputs.csv_input import CSVInput


class InputSelector(object):
    """ 样本数据集的访问适配器
    根据样本的数据存储类型，实例化对应的解析器获取数据流

    Args:
        sample_config: 样本配置
        feature_configs: 特征字典配置
    """

    def __init__(self, sample_config, feature_configs):
        self.sample_config = sample_config
        self.feature_configs = feature_configs

    def get_input(self):
        """ 获取对应数据类型的输入

        Returns:
            模型的输入
        """
        if self.sample_config.format == SampleFormat.TFRECORD:
            ins = TFRecordInput(sample_config=self.sample_config,
                                feature_configs=self.feature_configs)
        elif self.sample_config.format == SampleFormat.CSV:
            if self.sample_config.format_params.names is None:
                # Find out which io function to open the file
                file_io_fn = lambda filename: file_io.FileIO(filename, "r")
                if self.sample_config.compression_type is not None:
                    compression_type_value = tensor_util.constant_value(self.sample_config.compression_type)
                    if compression_type_value is None:
                        raise ValueError("Received unknown compression_type")
                    if compression_type_value == "GZIP":
                        file_io_fn = lambda filename: gzip.open(filename, "rt")
                    elif compression_type_value == "ZLIB":
                        raise ValueError(
                            "compression_type (%s) is not supported for probing columns" %
                            self.sample_config.compression_type)
                    elif compression_type_value != "":
                        raise ValueError("compression_type (%s) is not supported" %
                                         self.sample_config.compression_type)
                if not self.sample_config.format_params.header:
                    raise ValueError("Cannot infer column names without a header line.")
                if self.sample_config.mode != 'batch':
                    raise ValueError("Cannot infer column names without batch mode.")
                # If column names are not provided, infer from the header lines
                filenames = tf.io.gfile.glob(self.sample_config.path)
                if len(filenames) == 0:
                    raise ValueError("Cannot infer column names without input file.")
                column_names = _infer_column_names(filenames, self.sample_config.format_params.delim, True, file_io_fn)
                self.sample_config.format_params.names = column_names
            ins = CSVInput(sample_config=self.sample_config,
                           feature_configs=self.feature_configs)
        else:
            raise ValueError('Not supported sample format: %s' % self.sample_config.format)
        return ins.get_input()

    def get_input_fn(self):
        has_label = False
        for name, cfg in self.feature_configs.items():
            if cfg.is_label:
                has_label = True
                break

        def input_fn():
            iterator = tf.compat.v1.data.make_one_shot_iterator(self.get_input())
            if not has_label:
                return iterator.get_next()
            else:
                batch_features, batch_labels = iterator.get_next()
                return batch_features, batch_labels

        return input_fn
