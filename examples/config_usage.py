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
from smbrec.models import EstimatorModel


class DNN(EstimatorModel):
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
        embeded_inputs = {}
        columns = []
        for column in DNN.COLUMNS:
            """if column not in DNN.NUM_COLUMNS:
                encoded_input = smbrec.layers.VocabEncoder(
                    self.vocabs[column], begin_index=1, key_dtype='string')(inputs[column])
                embeded_inputs[column] = tf.keras.layers.Embedding(input_dim=len(self.vocabs[column])+1,
                                                                   output_dim=8)(encoded_input)
            else:
                embeded_inputs[column] = tf.keras.layers.experimental.preprocessing.CategoryEncoding(
                    num_tokens=len(self.vocabs[column])+1
                )(inputs[column])"""
            columns.append(tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_vocabulary_list(column, self.vocabs[column]), 4))
        stacked_inputs = tf.compat.v1.feature_column.input_layer(inputs, columns)
        #stacked_inputs = tf.concat(tf.nest.flatten(embeded_inputs), axis=1)
        output = tf.keras.layers.Dense(100, activation='relu')(stacked_inputs)
        output = tf.keras.layers.Dense(50, activation='relu')(output)
        output = tf.keras.layers.Dense(1)(output)
        return output

    def serving_outputs(self, logits):
        return tf.keras.layers.Activation('sigmoid')(logits)
