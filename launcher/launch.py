# -*- coding: UTF-8 -*-
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

from smbrec.utils.commandline_config import Config
from launcher.keras_launcher import KerasLauncher
from launcher.estimator_launcher import EstimatorLauncher
from utils import load_utils


if __name__ == '__main__':
    config = Config(sys.argv[1], name='SMBREC Pipeline Configuration')
    feature_configs = load_utils.load_feature_config(config)

    model = load_utils.load_model(config, feature_configs)
    if isinstance(model, tf.keras.Model):
        launcher = KerasLauncher(model=model, config=config, feature_configs=feature_configs)
    else:
        launcher = EstimatorLauncher(model=model, config=config, feature_configs=feature_configs)

    print("Configs:", config)
    print("Feature Configs:", feature_configs)

    launcher.run()
