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

import inspect
import os.path

import tensorflow as tf
from datetime import datetime, timedelta
from enum import Enum, unique
from smbrec.utils import tf_api


@unique
class DataType(Enum):
    INT = "int"
    FLOAT = "float"
    STRING = "string"


@unique
class FeatureFormat(Enum):
    KV = 'kv'
    LIST = 'list'
    VECTOR = 'vector'
    VAL = 'val'


@unique
class SampleFormat(Enum):
    CSV = "csv"
    TFRECORD = "tfrecord"


DATATYPE_2_TFTYPE = {
    DataType.INT: tf.int64,
    DataType.FLOAT: tf.float32,
    DataType.STRING: tf.string
}


# tensorflow支持的数据压缩类型
COMPRESSION_TYPES = set(['GZIP', 'ZLIB'])
# Sparse类型的FeatureFormat
SPARSE_FEATURE_FORMAT = set([FeatureFormat.KV, FeatureFormat.LIST])


def _value_2_enum(T, v):
    """ 通过enum的value获取enum值

    Args:
        T: enum类型
        v: enum的value

    Returns:
        v在enum中的值，找不到则抛出异常
    """
    if isinstance(v, T):
        return v
    elif v in T._value2member_map_:
        return T._value2member_map_[v]
    else:
        raise ValueError("Can't find %s in %s" % (v, str(T)))


class ConfigBase(dict):
    """ 配置基类，所有配置类都集成自该类。
    只允许在__init__函数中添加指定配置项，不允许通过setattr的方式添加配置项。

    Args:
        配置的配置项
    """

    def __init__(self, **kwargs):
        super(ConfigBase, self).__init__(**kwargs)
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    def __str__(self):
        attrs = vars(self)
        return ", ".join([attr+":"+str(attrs[attr]) for attr in attrs])

    def __setitem__(self, key, value):  # real signature unknown
        """ Set self[key] to value. """
        super(ConfigBase, self).__setitem__(key, value)
        setattr(self, key, value)

    def __delitem__(self, key):  # real signature unknown
        """ Delete self[key]. """
        super(ConfigBase, self).__delitem__(key)
        delattr(self, key)

    @classmethod
    def from_dict(cls, config):
        kwargs = {}
        fn_spec = inspect.getfullargspec(cls.__init__)
        for key, value in config.items():
            if fn_spec.varkw is None and key not in fn_spec.args:
                continue
            kwargs[key] = value
        return cls(**kwargs)

    def to_dict(self):
        # TODO(rabin): 最好在__new__函数中记录__init__函数的参数列表，参考tf-agent
        fn_spec = inspect.getfullargspec(self.__init__)
        return {[(param, value) for param, value in self.__dict__.items()
                 if param in fn_spec.args or fn_spec.varkw is None]}


class ConfigDictBase(dict):
    """ 配置集合类基类
    """
    key_attr = None     # 作为key的attr name
    value_type = None   # config值的类型

    def __init__(self, **kwargs):
        super(ConfigDictBase, self).__init__(**kwargs)

    def __str__(self):
        return ",".join(["%s: %s" % (k, str(v)) for k, v in self.items()])

    @classmethod
    def from_dict(cls, configs):
        """ 从字典构建配置集合

        Args:
            configs: 配置字典

        Returns:
            ConfigDict实例
        """
        instance = cls()
        for key, value in configs.items():
            instance[key] = cls.value_type.from_dict(value)
        return instance

    @classmethod
    def from_list(cls, configs):
        """ 从数组构建配置集合

        Args:
            configs: 配置数组

        Returns:
            ConfigDict实例
        """
        instance = cls()
        for config in configs:
            item = cls.value_type.from_dict(config)
            if cls.key_attr is None or not hasattr(item, cls.key_attr):
                raise ValueError("Don't have %s attribute." % cls.key_attr)
            attr_val = getattr(item, cls.key_attr)
            if attr_val in instance:
                raise KeyError("Duplicated item %s." % attr_val)
            instance[attr_val] = item
        return instance

    def is_valid(self, instance):
        """ 判断值是否合法

        Args:
            instance: 待验证的值

        Returns:
            True or False
        """
        if not isinstance(instance, type(self).value_type):
            return False
        return True

    def to_dict(self):
        params = {}
        for name, config in self.items():
            params[name] = config.to_dict()
        return params

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError("Can't find %s" % item)

    def __setattr__(self, attr, value):
        if not self.is_valid(value):
            raise ValueError("It's not a valid value %s." % value)
        self[attr] = value


class FeatureConfig(ConfigBase):
    """ 特征配置

    Args:
        name: 特征名
        is_label: 布尔值或整形，是否目标特征
        vocab_size: 词表大小，对ID类特征才有意义
        dimension: 特征值的维度，单值特征为0，例如：Embedding输入的embedding dim
        format: 特征存储的格式，支持 val，默认 val、list、vector、kv
        data_type: 特征的数据类型，支持 int、float、string，默认 int
        default: 默认值
    """

    DEFAULT_VALUE = {
        "int": 0,
        "float": 0.0,
        "string": ""
    }

    def __init__(self,
                 name,
                 is_label=False,
                 vocab_size=None,
                 dimension=0,
                 format='val',
                 data_type='int',
                 default=None):
        self.data_type = data_type
        default = FeatureConfig.DEFAULT_VALUE[self.data_type.value] if default is None else default
        super(FeatureConfig, self).__init__(name=name,
                                            is_label=is_label,
                                            vocab_size=vocab_size,
                                            dimension=dimension,
                                            format=format,
                                            default=default)

    @property
    def is_label(self):
        return self._is_label

    @is_label.setter
    def is_label(self, value):
        self._is_label = value if isinstance(value, bool) else value > 0

    @property
    def format(self):
        return self._format

    @format.setter
    def format(self, value):
        self._format = _value_2_enum(FeatureFormat, value)

    @property
    def data_type(self):
        return self._data_type

    @data_type.setter
    def data_type(self, value):
        self._data_type = _value_2_enum(DataType, value)

    def get_feature_spec(self):
        """ 获取当前特征的FeatureSpec

        Returns:
            FeatureSpec
        """
        default_value = self.default if self.dimension == 0 \
            else [[self.default] * self.dimension]
        dimension = [] if self.dimension == 0 else [self.dimension]
        if self.format in (FeatureFormat.VECTOR, FeatureFormat.VAL):
            feature_spec = tf.io.FixedLenFeature(dimension,
                                                 DATATYPE_2_TFTYPE[self.data_type],
                                                 default_value)
        elif self.format == FeatureFormat.LIST:
            feature_spec = tf.io.VarLenFeature(DATATYPE_2_TFTYPE[self.data_type])
        elif self.format == FeatureFormat.KV:
            # TODO(rabin) 对于KV特征，用vocab_size存储矩阵大小
            assert self.vocab_size is not None, "SparseFeature's vocab_size can't be None"
            feature_spec = tf.io.SparseFeature(index_key=self.name,
                                               value_key=self.name + ":weight",
                                               dtype=DATATYPE_2_TFTYPE[self.data_type],
                                               size=self.vocab_size)
        else:
            raise ValueError("Not supported feature format: %s" % self.format)

        return feature_spec


class FeatureConfigs(ConfigDictBase):
    key_attr = 'name'
    value_type = FeatureConfig

    def __init__(self, **kwargs):
        super(FeatureConfigs, self).__init__(**kwargs)

    def get_feature_spec(self, exclude_labels=False):
        feature_specs = {}
        for key, value in self.items():
            if exclude_labels and value.is_label:
                continue
            # TODO(rabin): TF1.14通过 label_name*count 的方式支持辅助loss
            if value.is_label and key.find('*') != -1:
                key = key.split('*')[0]
            feature_specs[key] = value.get_feature_spec()

        return feature_specs


class CSVFormatParams(ConfigBase):
    """ CSV格式样本的特定解析参数

    Args:
        names: 列名
        delim: 分隔符，默认为逗号","
        na_value: 识别为NaN的字符串
        header: 布尔值或整形，指示文件第一行是否为列名
        select_columns: 需要解析的列序号，默认为None，表示都解析
        ignore_errors: 是否忽略解析错误的行
    """

    def __init__(self,
                 names=None,
                 delim=',',
                 na_value='',
                 header=False,
                 select_columns=None,
                 ignore_errors=True):
        super(CSVFormatParams, self).__init__(names=names,
                                              delim=delim,
                                              na_value=na_value,
                                              header=header,
                                              select_columns=select_columns,
                                              ignore_errors=ignore_errors)

    @property
    def names(self):
        return self._names

    @names.setter
    def names(self, value):
        self._names = value.split(self.delim) if isinstance(value, str) else value

    @property
    def select_columns(self):
        if not self._all_select_columns_index:
            column_indexes = []
            for column in self._select_columns:
                index = self._names.index(column)
                column_indexes.append(index)
            self._select_columns = column_indexes
            self._all_select_columns_index = True

        return self._select_columns

    @select_columns.setter
    def select_columns(self, value):
        self._all_select_columns_index = True
        self._select_columns = value

        if self._select_columns is not None:
            for column in self._select_columns:
                if not isinstance(column, int):
                    self._all_select_columns_index = False
                    break

    @property
    def header(self):
        return self._header

    @header.setter
    def header(self, value):
        self._header = value if isinstance(value, bool) else value > 0


class StreamingParams(ConfigBase):
    """ 流样本的配置参数

    Args:
        buffer_size: 文件队列的缓存大小
        checkpoint: 断点恢复标识文件路径
        start_point: 启动的数据时间
        timeout: 数据消费的超时时间
        time_fmt: 时间格式化字符串
        ready_flag: 目录就绪的标志文件
    """

    def __init__(self,
                 buffer_size=100,
                 checkpoint=None,
                 start_point=None,
                 timeout=5,
                 time_fmt="%Y%m%d %H:%M",
                 ready_flag=None):
        super(StreamingParams, self).__init__(buffer_size=buffer_size,
                                              checkpoint=checkpoint,
                                              start_point=start_point,
                                              timeout=timeout,
                                              time_fmt=time_fmt,
                                              ready_flag=ready_flag)


class SampleConfig(ConfigBase):
    """ 样本配置

    Args:
        format: 样本的存储格式
        size: 样本量
        path: 样本路径
        cache_path: 缓存路径，有缓存路径优先使用缓存路径
        days: 样本天数，>1时，path根目录必须是 %Y%m%d，默认1
        batch_size: batch的大小，默认512
        shuffle: 是否shuffle，默认False
        compression_type: 数据集的压缩类型，可取值有 ZLIB、GZIP
        num_parallel_reads: DataSet并行读数据文件的并行数，默认为 None，表示不并行
        num_parallel_calls: DataSet并行处理数据的并行度，默认为 tf.data.AUTOTUNE，表示取可用cpu数
        drop_remainder: 是否丢掉不足batch size的最后一个batch，train时建议开启，避免最后一个batch波动
        epochs: 样本重复次数，默认为1
        format_params: 样本格式的解析参数
        mode: 数据流形式，batch or streaming，默认batch
        streaming_params: 流式数据配置
        remove_labels: 是否从inputs移除labels，默认True
    """

    def __init__(self,
                 format=None,
                 size=None,
                 path=None,
                 cache_path=None,
                 days=1,
                 batch_size=512,
                 shuffle=False,
                 compression_type=None,
                 num_parallel_reads=None,
                 num_parallel_calls=tf_api.AUTOTUNE,
                 drop_remainder=False,
                 epochs=1,
                 format_params=None,
                 mode="batch",
                 streaming_params=None,
                 remove_labels=True):
        assert compression_type is None or compression_type in COMPRESSION_TYPES
        super(SampleConfig, self).__init__(format=format,
                                           size=size,
                                           path=path,
                                           cache_path=cache_path,
                                           days=days,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           compression_type=compression_type,
                                           num_parallel_reads=num_parallel_reads,
                                           num_parallel_calls=num_parallel_calls,
                                           drop_remainder=drop_remainder,
                                           epochs=epochs,
                                           format_params=format_params,
                                           mode=mode,
                                           streaming_params=streaming_params,
                                           remove_labels=remove_labels)

    @property
    def path(self):
        """ 获取数据路径

        Returns:
            如果是多天，返回分天的目录数组
        """
        path = self.cache_path or self._path
        if self.days > 1 and isinstance(path, str):
            final_path = []
            dir_name = os.path.dirname(path)
            end_date = datetime.strptime(os.path.basename(path), '%Y%m%d')
            for i in range(self.days):
                date = (end_date - timedelta(days=i)).strftime('%Y%m%d')
                final_path.append(os.path.join(dir_name, date))
            return final_path[::-1]
        else:
            return path

    @path.setter
    def path(self, value):
        self._path = value

    @property
    def format(self):
        return self._format

    @format.setter
    def format(self, value):
        self._format = _value_2_enum(SampleFormat, value)

    @property
    def format_params(self):
        return self._format_params

    @format_params.setter
    def format_params(self, value):
        if isinstance(value, dict) and self.format == SampleFormat.CSV:
            self._format_params = CSVFormatParams.from_dict(value)
        else:
            self._format_params = value

    @property
    def streaming_params(self):
        return self._streaming_params

    @streaming_params.setter
    def streaming_params(self, value):
        if isinstance(value, dict):
            self._streaming_params = StreamingParams.from_dict(value)
        else:
            self._streaming_params = value

    @property
    def remove_labels(self):
        return self._remove_labels

    @remove_labels.setter
    def remove_labels(self, value):
        self._remove_labels = value if isinstance(value, bool) else value > 0

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, value):
        self._shuffle = value if isinstance(value, bool) else value > 0

    @property
    def drop_remainder(self):
        return self._drop_remainder

    @drop_remainder.setter
    def drop_remainder(self, value):
        self._drop_remainder = value if isinstance(value, bool) else value > 0

