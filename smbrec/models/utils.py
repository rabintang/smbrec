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

import sys
import tensorflow as tf
import smbrec.metrics
from smbrec.utils.commandline_config import Config


def initialize_optimizer(config):
    """ 获取optimizer实例
    如果配置文件无法满足，可以重写该类方法，返回optimizer实例

    Args:
        config: optimizer config

    Returns:
        optimizer实例
    """
    optimizer_params = {}
    if isinstance(config, Config):
        optimizer_method = config.method
        optimizer_params = config.params.preset_config if "params" in config else {}
    else:
        optimizer_method = config

    pos = optimizer_method.rfind('.')
    if pos == -1:
        return optimizer_method
    model_module = optimizer_method[:pos]
    model_class = optimizer_method[pos + 1:]
    module = model_module
    __import__(module)
    return getattr(sys.modules[module], model_class)(**optimizer_params)


def get_metric(name):
    """ 根据metric的名字获取metric方法

    Args:
        name: metric名字

    Returns:
        metric方法
    """
    if hasattr(smbrec.metrics, name):
        return getattr(smbrec.metrics, name)()
    else:
        return name


def initialize_model(clazz, config, feature_configs):
    """ 通过配置实例化当前模型

    Args:
        clazz: 模型类
        config: pipeline配置
        feature_configs: 特征配置字典

    Returns:
        当前模型实例
    """
    # prepare optimizer
    optimizer = None if 'optimizer' not in config.train_params else \
        initialize_optimizer(config.train_params.optimizer)

    # prepare loss
    if isinstance(config.train_params.loss, list):
        loss = {}
        loss_weights = {}
        for value in config.train_params.loss:
            loss[value['label']] = value['loss']
            loss_weights[value['label']] = value['weight']
    else:
        loss = config.train_params.loss
        loss_weights = None

    # prepare metrics
    eval_config = config.get('eval_params', Config({}, name='dict eval_params', read_command_line=False))
    metrics = None
    if 'metrics' in eval_config:
        if isinstance(eval_config.metrics, Config):
            metrics = {}
            for name, value in eval_config.metrics.preset_config.items():
                metrics[name] = [get_metric(item) for item in value]
        else:
            metrics = [get_metric(value) for value in eval_config.metrics]

    model_config = config.get(
        "model_params", Config({}, name='dict model_params', read_command_line=False))
    model = clazz(feature_configs=feature_configs, model_config=model_config, config=config)

    model.compile(optimizer=optimizer,
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=metrics)
    return model
