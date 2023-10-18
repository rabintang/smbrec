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

import abc
import tensorflow as tf
from random import shuffle
from smbrec.inputs.streaming_generator import StreamingGenerator
from smbrec.utils import file_utils


class Input(object):
    """ 样本数据集的访问接口
    给定样本和特征配置，获取对应的 tf.data.Dataset 对象

    Args:
        sample_config: 样本配置
        feature_configs: 特征字典配置
    """

    def __init__(self, sample_config, feature_configs):
        self.sample_config = sample_config
        self.feature_configs = feature_configs
        print('Sample Config:', self.sample_config)

    def get_input(self):
        """ 构造数据集对应的模型输入

        Returns:
            模型的输入
        """
        if self.sample_config.mode == "batch":
            try:
                if isinstance(self.sample_config.path, str):
                    data_files = file_utils.listdir(self.sample_config.path)
                else:
                    data_files = []
                    for path in self.sample_config.path:
                        data_files.extend(file_utils.listdir(path))
                # TODO(rabin) 对数据文件进行打散
                if self.sample_config.shuffle:
                    shuffle(data_files)
                with open('logs/train_files', 'w') as ouf:
                    ouf.write('\n'.join(data_files))
                dataset = tf.data.Dataset.from_tensor_slices(data_files)
            except IOError as e:
                raise RuntimeError('List data file fail: %s, msg: %s' % (self.sample_config.path, str(e)))
        elif self.sample_config.mode == "streaming":
            generator = StreamingGenerator(self.sample_config)
            generator.start()
            dataset = tf.data.Dataset.from_generator(
                lambda: generator,
                output_signature=tf.TensorSpec(shape=[], dtype=tf.string))
        else:
            raise ValueError('Not supported data mode: %s' % self.sample_config.mode)

        return self._create_input(dataset)

    def _get_label(self, features, label):
        return features.pop(label) if self.sample_config.remove_labels else features[label]

    @abc.abstractmethod
    def _create_input(self, dataset):
        raise NotImplementedError
