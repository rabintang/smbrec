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

import tensorflow as tf
import time

from smbrec.datasets import FakeData
from smbrec.models import Model


class DNN(Model):
    def __init__(self, feature_cfgs):
        super(DNN, self).__init__(feature_cfgs=feature_cfgs)

    def forward(self, inputs=None):
        input = [value for name, value in inputs.items() if name.startswith('F')]
        input = tf.keras.layers.Concatenate()(input)
        task1 = tf.keras.layers.Dense(16, activation='relu', use_bias=True)(input)
        task2 = tf.keras.layers.Dense(16, activation='relu', use_bias=True)(input)
        output1 = tf.keras.layers.Dense(1, use_bias=False, name='label1')(task1)
        output2 = tf.keras.layers.Dense(1, use_bias=False, name='label2')(task2)
        return {'label1': output1, 'label2': output2}

    def serving_outputs(self, logits):
        """ 模型离线导出时定义score，也可以放到模型服务进行计算，则serving将返回两个值 """
        return logits['label1'] * logits['label2']


if __name__ == "__main__":
    """
    https://colab.research.google.com/gist/goldiegadde/0ad3fed10b0cbfb3633ee3ff05dcad31/github-issue-34691.ipynb#scrollTo=DdD37NIQxfvb
    https://github.com/tensorflow/tensorflow/issues/34691
    https://github.com/tensorflow/tensorflow/issues/34114
    """

    fake_data = FakeData(sample_size=1000, label_num=2, float_num=10, int_num=10)

    # 4.Define Model,train,predict and evaluate
    model = DNN(fake_data.feature_cfgs)
    model.summary()
    model.compile('adam', loss={'label1': 'binary_crossentropy', 'label2': 'binary_crossentropy'},
                  loss_weights={'label1': 0.4, 'label2': 0.6},
                  metrics={'label1': 'mean_absolute_error', 'label2': 'mean_absolute_error'})

    history = model.fit(fake_data.get_input(), verbose=2)
    inputs = {key: tf.TensorSpec(shape=value.shape, dtype=value.dtype, name=key)
              for key, value in model.input.items()}
    saved_path = './tmp/saved_models/multi_task/' + str(int(time.time()))
    model.save(saved_path, signatures=model.signature_function.get_concrete_function(
        inputs=inputs
    ))
