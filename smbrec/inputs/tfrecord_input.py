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
from collections import Iterable
from functools import partial

from smbrec.inputs.input import Input
from smbrec.utils import tf_api


class TFRecordInput(Input):
    """ 样本数据集的访问接口
    给定样本和特征配置，获取对应的 tf.data.Dataset 对象

    Args:
        sample_config: 样本配置
        feature_configs: 特征字典配置
    """

    def __init__(self, sample_config, feature_configs):
        super(TFRecordInput, self).__init__(sample_config=sample_config,
                                            feature_configs=feature_configs)

    def _create_input(self, dataset):
        """ TFRecords格式数据的访问接口

        Args:
            dataset: 封装了文件列表的DataSet

        Returns:
            tf.data.Dataset
        """
        def _parse_fn(example_proto, feature_spec, labels):
            features = tf.io.parse_example(example_proto, feature_spec)
            if labels is not None:
                if len(labels) == 1:
                    # TODO(rabin): 1.14辅助loss需要显式指定多个label，通过采用 label_name*count 的形式指定个数
                    label, count = labels[0], 1
                    if labels[0].find("*") != -1:
                        label, count = labels[0].split("*")
                        count = int(count)
                    targets = self._get_label(features, label)
                    if count > 1:
                        targets = tuple([targets] * count)
                else:
                    targets = {label: self._get_label(features, label) for label in labels}
                return features, targets
            else:
                return features

        # NOTE(rabin) dataset.interleave 接口将导致效率显著下降，原因未查明
        dataset = dataset.apply(
            tf.data.experimental.parallel_interleave(
                lambda filename: tf.data.TFRecordDataset(
                    filename,
                    compression_type=self.sample_config.compression_type,
                    buffer_size=self.sample_config.batch_size * 20
                ),
                sloppy=True,
                block_length=1,
                cycle_length=16
            )
        )
        dataset = dataset.prefetch(buffer_size=4 << 13).repeat(self.sample_config.epochs)

        # NOTE(rabin) shuffle会导致from_generator假死，原因未明
        if self.sample_config.shuffle and self.sample_config.mode != 'streaming':
            dataset = dataset.shuffle(buffer_size=100 * self.sample_config.batch_size)
        if tf.__version__ < "2.5.0":
            dataset = dataset.batch(batch_size=self.sample_config.batch_size,
                                    drop_remainder=self.sample_config.drop_remainder)
        else:
            dataset = dataset.batch(batch_size=self.sample_config.batch_size,
                                    drop_remainder=self.sample_config.drop_remainder,
                                    num_parallel_calls=self.sample_config.num_parallel_calls)
        feature_spec = self.feature_configs.get_feature_spec()
        labels = [name for name, cfg in self.feature_configs.items() if cfg.is_label]
        dataset = dataset.map(partial(_parse_fn, feature_spec=feature_spec, labels=labels),
                              num_parallel_calls=self.sample_config.num_parallel_calls)
        dataset = dataset.prefetch(tf_api.AUTOTUNE)
        return dataset
