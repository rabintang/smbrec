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

from smbrec.inputs import InputAdapter
from smbrec.core.config import SampleConfig, FeatureConfigs, FeatureConfig, SampleFormat, CSVFormatParams, DataType


class DNN(tf.keras.Model):
    def __init__(self, feature_cfgs, **kwargs):
        self.feature_cfgs = feature_cfgs
        self._build(**kwargs)

    def forward(self, inputs=None):
        input = [value for name, value in inputs.items() if name.startswith('I')]
        input = tf.keras.layers.Concatenate()(input)
        print(input.shape)
        return tf.keras.layers.Dense(1, use_bias=True, activation='sigmoid')(input)

    def _build(self, **kwargs):
        """
        Replicates Model(inputs=[inputs], outputs=[outputs]) of functional model.
        """
        # Replace with shape=[None, None, None, 1] if input_shape is unknown.
        inputs = {
            'I'+str(i): tf.keras.Input(shape=[1], dtype=tf.float32)
            for i in range(1, 14)
        }
        outputs = self.forward(inputs)
        super(DNN, self).__init__(name="DNN", inputs=inputs, outputs=outputs, **kwargs)


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
    input_adapter = InputAdapter(sample_cfg=sample_cfg, feature_cfgs=feature_cfgs)

    # 4.Define Model,train,predict and evaluate
    model = DNN(feature_cfgs)
    model.summary()
    model.compile("adam", "binary_crossentropy",
                  metrics=['AUC'])

    history = model.fit(input_adapter.get_input(), epochs=10)
