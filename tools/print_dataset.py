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

import numpy as np
import sys
import tensorflow as tf
from smbrec.utils.commandline_config import Config
from utils import load_utils


def print_data(input_selector):
    """ 预测数据

    Args:
        input_selector: 输入数据
    """
    for batch in input_selector.get_input():
        for key, tensor in batch[0].items():
            print(key, tensor)
        break


if __name__ == '__main__':
    config = Config(sys.argv[1], name='SMBREC Pipeline Configuration')
    feature_configs = load_utils.load_feature_config(config)
    assert 'data' in config.eval_params
    print("Configs:", config)
    print("Feature Configs:", feature_configs)

    load_utils.set_environ()
    load_utils.set_devices(config.train_params.get('gpus', ''))

    input_selector = load_utils.get_input_selector(feature_configs, config.eval_params.data)
    print_data(input_selector)
