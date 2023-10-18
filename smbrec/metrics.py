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
from tensorflow.keras.metrics import Metric
from tensorflow.python.keras.metrics import MeanMetricWrapper


class OE(Metric):
    """ 预估值与真实值偏差OE(Observation Over Expectation)

    该指标计算真实值比预估值的偏差，即 sum(真实值)/sum(预估值)。计算
    预估值相比真实值的偏离服务。

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
    """

    def __init__(self, name='OE', dtype=None):
        super(OE, self).__init__(name=name, dtype=dtype)
        self.true_sum = self.add_weight(
            'true_sum', initializer=tf.zeros_initializer)
        self.pred_sum = self.add_weight(
            'pred_sum', initializer=tf.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            y_true = tf.multiply(y_true, sample_weight)
            y_pred = tf.multiply(y_pred, sample_weight)
        update_true_op = self.true_sum.assign_add(tf.reduce_sum(y_true))
        with tf.control_dependencies([update_true_op]):
            return self.pred_sum.assign_add(tf.reduce_sum(y_pred))

    def result(self):
        return tf.math.divide_no_nan(self.true_sum, self.pred_sum)


def pred_mean(y_true, y_pred):
  return y_pred


class Mean(MeanMetricWrapper):
    """ 预估值均值

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
    """

    def __init__(self, name='Mean', dtype=None):
        super(Mean, self).__init__(pred_mean, name, dtype=dtype)

