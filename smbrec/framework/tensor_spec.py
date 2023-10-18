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


class NamedSparseTensorSpec(tf.SparseTensorSpec):
    """ 命名的SparseTensorSpec

    SparseTensorSpec没有name参数，导致无法通过原始特征名构建serving的输入以及模型推理。
    这里重载SparseTensorSpec，对其三个TensorSpec进行命名

    """

    def __init__(self, shape=None, dtype=tf.float32, name=None):
        super(NamedSparseTensorSpec, self).__init__(shape, dtype)
        self.name = name

    @property
    def _component_specs(self):
        if not self.name:
            return super(NamedSparseTensorSpec, self)._component_specs()
        rank = self._shape.ndims
        num_values = None
        return [
            tf.TensorSpec([num_values, rank], tf.int64, name=self.name + "_indices"),
            tf.TensorSpec([num_values], self._dtype, name=self.name + "_values"),
            tf.TensorSpec([rank], tf.int64, name=self.name + "_dense_shape")
        ]


