# -*- coding: UTF-8 -*-
#  Copyright (c) 2023, The SmbRec Authors.  All rights reserved.
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
import os
import sys
import tensorflow as tf

from smbrec.core.config import FeatureConfigs, SampleConfig
from smbrec.inputs import InputSelector
from smbrec.models import utils as model_utils
from smbrec.utils import hdfs_utils, file_utils


def load_model(config, feature_configs):
    """ 加载模型

    Args:
        config: pipeline config
        feature_configs: 特征配置

    Returns:
        模型
    """
    model_class = config.model_clazz
    pos = model_class.rfind('.')
    model_module = model_class[:pos]
    model_class = model_class[pos + 1:]
    module = model_module
    __import__(module)
    model = model_utils.initialize_model(clazz=getattr(sys.modules[module], model_class),
                                         feature_configs=feature_configs,
                                         config=config)
    if isinstance(model, tf.keras.Model):
        model._assert_compile_was_called()
    return model


def load_feature_config(config):
    """ 加载特征配置文件

    Args:
        config: pipeline config

    Returns:
        FeatureConfigs实例
    """
    HDFS_ROOT = "hdfs://hdfs2-nameservice/data/pangu/config/scene/"

    if "feature_config_path" in config:
        feature_configs = json.load(open(config.feature_config_path))
    elif "scene" in config and "sample_name" in config:
        current_config_path = os.path.join(HDFS_ROOT, config.scene, "current.json")
        # 优先获取指定的样本版本，否则读取最新版
        if "sample_version" in config:
            sample_config_path = os.path.join(HDFS_ROOT, config.scene, "sample",
                                              config.sample_name, config.sample_version + ".json")
        else:
            sample_config_dir = os.path.join(HDFS_ROOT, config.scene, "sample", config.sample_name)
            sample_versions = hdfs_utils.listdir(sample_config_dir)
            sample_version = sorted(sample_versions)[-1]
            sample_config_path = os.path.join(sample_config_dir, sample_version + ".json")

        current_config = json.loads(hdfs_utils.readall(current_config_path))
        sample_config = json.loads(hdfs_utils.readall(sample_config_path))
        feature_config_map = {item["name"]: item for item in current_config["features"]}
        feature_configs = [feature_config_map[feat] for feat in sample_config["features"]
                           + sample_config["original_features"]]
        for source_config in sample_config.get("sources", []):
            for label_config in source_config['labels']:
                feature_configs.append({
                    "name": label_config["name"],
                    "is_label": 1,
                    "format": "val",
                    "data_type": "float"
                })
    else:
        raise AttributeError("Cannot find feature configuration.")
    return FeatureConfigs.from_list(feature_configs)


def get_input_selector(feature_configs, data_config):
    sample_config = SampleConfig.from_dict(data_config)
    input_selector = InputSelector(sample_config=sample_config, feature_configs=feature_configs)
    return input_selector


def load_weights(model, model_path):
    """ 预测与评估数据

    Args:
        model: 模型
        model_path: 模型文件路径
    Returns:
        模型
    """
    if isinstance(model, tf.keras.models.Model):
        model.load_weights(model_path)
    else:
        model.checkpoint_dir = model_path
        model.recompile()
    return model


def set_environ():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['KMP_BLOCKTIME'] = '1'
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
    os.environ['KMP_SETTINGS'] = 'TRUE'


def set_devices(gpu_indices):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(index) for index in gpu_indices])

