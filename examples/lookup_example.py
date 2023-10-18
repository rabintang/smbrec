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
import time

import tensorflow as tf
import pandas as pd
import smbrec.layers
from smbrec.inputs import InputAdapter
from smbrec.models import Model
from smbrec.core.config import SampleConfig, FeatureConfigs, FeatureConfig, SampleFormat, CSVFormatParams, DataType


class DNN(Model):
    def __init__(self, feature_cfgs, model_params):
        super(DNN, self).__init__(feature_cfgs=feature_cfgs, model_params=model_params)

    def forward(self, inputs=None):
        input = [value for name, value in inputs.items() if name.startswith('I')]
        c1 = smbrec.layers.VocabEncode(self.model_params['c1'])(inputs['C1'])
        c2 = smbrec.layers.VocabEncode(self.model_params['c2'])(inputs['C2'])
        #input.append(tf.squeeze(smbrec.layers.Embedding(1000, 16)(c1), axis=1))
        #input.append(tf.squeeze(smbrec.layers.Embedding(1000, 16)(c2), axis=1))
        input.append(tf.squeeze(tf.keras.layers.Embedding(1000, 16)(c1), axis=1))
        input.append(tf.squeeze(tf.keras.layers.Embedding(1000, 16)(c2), axis=1))
        input = tf.keras.layers.Concatenate()(input)
        input = tf.keras.layers.Dense(16, activation='relu')(input)
        return tf.keras.layers.Dense(1, use_bias=True)(input)


if __name__ == "__main__":
    df = pd.read_csv('./examples/data/criteo/criteo_sample.txt', sep=',')
    c1 = df['C1'].unique()
    c2 = df['C2'].unique()
    with open('c1_vocab.txt', 'w') as writer:
        writer.write('\n'.join(c1))
    model_params = {
        'c1': 'c1_vocab.txt',
        'c2': c2
    }

    select_columns = ['label'] + ['I'+str(i) for i in range(1, 14)] + ['C'+str(i) for i in range(1, 3)]
    sample_cfg = SampleConfig(path='./examples/data/criteo/criteo_sample.txt',
                              format=SampleFormat.CSV,
                              shuffle=True,
                              epochs=10000,
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
    for i in range(1, 3):
        feature_cfgs['C'+str(i)] = FeatureConfig(name='C'+str(i), data_type=DataType.STRING)
    input_adapter = InputAdapter(sample_cfg=sample_cfg, feature_cfgs=feature_cfgs)

    # 4.Define Model,train,predict and evaluate
    model = DNN(feature_cfgs, model_params)
    model.summary()
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'])

    start = time.time()
    history = model.fit(input_adapter.get_input(), verbose=2)
    end = time.time()
    print(end - start)
    model.save('./tmp/serving/' + str(int(time.time())))
