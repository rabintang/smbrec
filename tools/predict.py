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


def predict(model, input_selector, output_file):
    """ 预测数据

    Args:
        model: 模型
        input_selector: 输入数据
        output_file: 预测结果保持路径
    """
    if isinstance(model, tf.keras.models.Model):
        input = input_selector.get_input()
    else:
        input = input_selector.get_input_fn()

    with open(output_file, 'w') as ouf:
        for idx, prediction in enumerate(model.predict(input)):
            if idx == 0:
                ouf.write('|'.join(prediction.keys()) + "\n")
            if idx % 10000 == 0 and idx != 0:
                print('finish: ', idx)
            ouf.write('|'.join([','.join([str(v) for v in val]) if type(val) in (list, tuple, np.ndarray) \
                      else str(val) for val in prediction.values()]) + "\n")
        print('total: ', idx)


if __name__ == '__main__':
    config = Config(sys.argv[1], name='SMBREC Pipeline Configuration')
    feature_configs = load_utils.load_feature_config(config)
    assert 'model_path' in config.train_params
    assert 'data' in config.eval_params
    assert 'predictions_path' in config.eval_params
    print("Configs:", config)
    print("Feature Configs:", feature_configs)

    load_utils.set_environ()
    load_utils.set_devices(config.train_params.get('gpus', ''))

    model = load_utils.load_model(config, feature_configs)
    input_selector = load_utils.get_input_selector(feature_configs, config.eval_params.data)
    load_utils.load_weights(model, config.train_params.model_path)
    predict(model, input_selector, config.eval_params.predictions_path)
