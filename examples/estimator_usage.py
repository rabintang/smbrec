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

import os
import tensorflow as tf
import timeit

from smbrec.inputs import InputAdapter
from smbrec.models import EstimatorModel
from smbrec.core.config import SampleConfig, FeatureConfigs, FeatureConfig, SampleFormat, CSVFormatParams, DataType


class DNN(EstimatorModel):
    def __init__(self, feature_confs, model_conf=None):
        super(DNN, self).__init__(feature_confs=feature_confs, model_conf=model_conf)

    def forward(self, inputs=None):
        input = [tf.expand_dims(value, axis=1) for name, value in inputs.items() if name.startswith('I')]
        input = tf.keras.layers.Concatenate(axis=-1)(input)
        input = tf.keras.layers.Dense(16, activation='relu')(input)
        return tf.keras.layers.Dense(1, use_bias=True)(input)

    def serving_outputs(self, logits):
        return tf.keras.layers.Activation('sigmoid')(logits)


if __name__ == "__main__":

    select_columns = ['label'] + ['I'+str(i) for i in range(1, 14)]
    root_dir = os.path.dirname(os.path.abspath(__file__))
    sample_cfg = SampleConfig(path=os.path.join(root_dir, 'data/criteo/criteo_sample.txt'),
                              format=SampleFormat.CSV,
                              shuffle=True,
                              format_params=CSVFormatParams(
                                  names=['label'] + ['I'+str(i) for i in range(14)] + ['C'+str(i) for i in range(27)],
                                  header=True,
                                  select_columns=select_columns)
                              )
    feature_cfgs = FeatureConfigs(
        label=FeatureConfig(name='label', data_type='float', is_label=True)
    )
    for i in range(1, 14):
        feature_cfgs['I'+str(i)] = FeatureConfig(name='I'+str(i), data_type=DataType.FLOAT)

    # 4.Define Model,train,predict and evaluate
    model = DNN(feature_cfgs)
    model.compile("Adam", "binary_crossentropy",
                  metrics=['auc'])
    input_adapter = InputAdapter(sample_conf=sample_cfg, feature_confs=feature_cfgs)

    print("time consuming:", timeit.timeit(lambda: model.fit(input_adapter.get_input_fn()), number=10))
    print(model.evaluate(input_adapter.get_input_fn()))
