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

import collections
import functools
import tensorflow as tf

from tensorflow.python.data.experimental.ops import error_ops
from tensorflow.python.data.ops import dataset_ops

from smbrec.core.config import CSVFormatParams
from smbrec.inputs.input import Input
from smbrec.utils import tf_api


class CSVInput(Input):
    """ 样本数据集的访问接口
    给定样本和特征配置，获取对应的 tf.data.Dataset 对象

    Args:
        sample_config: 样本配置
        feature_configs: 特征字典配置
    """

    def __init__(self, sample_config, feature_configs):
        super(CSVInput, self).__init__(sample_config=sample_config,
                                       feature_configs=feature_configs)

    def _create_input(self, dataset):
        """ CSV格式数据的访问接口

        Args:
            dataset: 封装了文件列表的DataSet

        Returns:
            tf.data.Dataset
        """
        def _to_dataset(filename):
            dataset = tf.data.experimental.CsvDataset(
                filename,
                record_defaults=defaults,
                field_delim=params.delim,
                use_quote_delim=True,
                na_value=params.na_value,
                select_cols=params.select_columns,
                header=params.header,
                compression_type=self.sample_config.compression_type,
                buffer_size=self.sample_config.batch_size * 20
            )
            if params.ignore_errors:
                dataset = dataset.apply(error_ops.ignore_errors())
            return dataset

        def _map_fn(*columns):
            features = collections.OrderedDict(zip(column_names, columns))
            if label_names:
                labels = {}
                if isinstance(label_names, str):
                    labels = self._get_label(features, label_names)
                else:
                    labels = {label: self._get_label(features, label) for label in labels}
                return features, labels
            return features

        params = self.sample_config.format_params or CSVFormatParams()
        label_names = [name for name, cfg in self.feature_configs.items() if cfg.is_label]
        label_names = label_names[0] if len(label_names) == 1 else label_names
        defaults = None
        column_names = params.names
        if params.select_columns is None:
            defaults = [self.feature_configs[name].default for name in params.names]
        else:
            defaults = [self.feature_configs[params.names[index]].default for index in params.select_columns]
        if params.select_columns is not None:
            # Pick the relevant subset of column names
            column_names = [params.names[i] for i in params.select_columns]
        # NOTE(rabin) dataset.interleave 接口将导致效率显著下降，原因未查明
        dataset = dataset.apply(
            tf.data.experimental.parallel_interleave(
                _to_dataset,
                sloppy=True,
                block_length=1,
                cycle_length=16
            )
        )
        dataset = dataset.prefetch(buffer_size=4 << 24).repeat(self.sample_config.epochs)
        # NOTE(rabin) shuffle会导致from_generator假死，原因未明
        if self.sample_config.shuffle and self.sample_config.mode != 'streaming':
            dataset = dataset.shuffle(buffer_size=100 * self.sample_config.batch_size)

        dataset = dataset.batch(batch_size=self.sample_config.batch_size,
                                drop_remainder=self.sample_config.drop_remainder)

        dataset = dataset_ops.MapDataset(
            dataset, _map_fn,
            use_inter_op_parallelism=False)
        dataset = dataset.prefetch(tf_api.AUTOTUNE)
        return dataset
