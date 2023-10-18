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

import os
import smbrec
import pandas as pd
import tensorflow as tf
from smbrec.models import KerasModel


class DNN(KerasModel):
    COLUMNS = [
        'age', 'workclass', 'fnlgwt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week', 'country'
    ]
    NUM_COLUMNS = [
        'age', 'fnlgwt', 'education_num', 'capital_gain', 'capital_loss',
        'hours_per_week'
    ]

    NUM_OOV_BUCKETS = 1

    def __init__(self, feature_configs, model_config=None, **kwargs):
        super(DNN, self).__init__(feature_configs=feature_configs, model_config=model_config, **kwargs)
        dframe = pd.read_csv(self.config.train_params.data.path, sep=',')
        print(dframe.head())
        self.vocabs = {}
        for column in DNN.COLUMNS:
            self.vocabs[column] = dframe[column].dropna().unique().tolist()

    def forward(self, inputs=None):
        embeded_inputs = []
        for column in DNN.COLUMNS:
            encoded_input = smbrec.layers.VocabEncoder(
                self.vocabs[column], begin_index=1,
                key_dtype='int64' if column in DNN.NUM_COLUMNS else 'string')(inputs[column])
            embeded_inputs.append(tf.squeeze(tf.keras.layers.Embedding(input_dim=len(self.vocabs[column])+1,
                                                                       output_dim=8)(encoded_input), axis=1))
        stacked_inputs = tf.keras.layers.Concatenate()(embeded_inputs)
        output = tf.keras.layers.Dense(100, activation='relu')(stacked_inputs)
        output = tf.keras.layers.Dense(50, activation='relu')(output)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(output)
        return output

    def get_config(self):
        config = super(DNN, self).get_config()
        config.update({
            'vocabs': self.vocabs
        })
        return config
