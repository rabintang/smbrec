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
import tensorflow as tf
from smbrec.core.config import FeatureConfig


class FakeData(object):
    def __init__(self,
                 sample_size=100,
                 label_num=1,
                 float_num=0,
                 int_num=0,
                 string_num=0):
        self.sample_size = sample_size
        self.label_num = label_num
        self.float_num = float_num
        self.int_num = int_num
        self.string_num = string_num

    def get_input(self):
        features = {}
        if self.label_num > 1:
            targets = {}
            for i in range(1, self.label_num+1):
                targets['label' + str(i)] = np.random.randint(0, 2, size=(self.sample_size, 1))
        else:
            targets = np.random.randint(0, 2, size=(self.sample_size, 1))
        for i in range(1, self.float_num+1):
            features['F' + str(i)] = np.random.random(size=(self.sample_size, 1))
        for i in range(1, self.int_num+1):
            features['I' + str(i)] = np.random.randint(1, 10000, size=(self.sample_size, 1))
        features_dataset = tf.data.Dataset.from_tensor_slices(features)
        labels_dataset = tf.data.Dataset.from_tensor_slices(targets)
        dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))
        return dataset

    @property
    def feature_cfgs(self):
        cfgs = {}
        for i in range(1, self.label_num+1):
            cfgs['label' + str(i)] = FeatureConfig(name='label' + str(i),
                                                   is_label=True,
                                                   dtype='int')
        for i in range(1, self.float_num + 1):
            cfgs['F' + str(i)] = FeatureConfig(name='F' + str(i),
                                               is_label=False,
                                               dtype='float')
        for i in range(1, self.int_num + 1):
            cfgs['I' + str(i)] = FeatureConfig(name='I' + str(i),
                                               is_label=False,
                                               dtype='int')

        return cfgs
