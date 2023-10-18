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

import functools

import numpy as np
import tensorflow as tf

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']
LABEL_COLUMN = 'survived'
LABELS = [0, 1]


def get_dataset(file_path):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=12,  # 为了示例更容易展示，手动设置较小的值
      label_name=LABEL_COLUMN,
      column_names=CSV_COLUMNS,
      na_value="?",
      num_epochs=1,
      ignore_errors=True)
  return dataset


raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)

examples, labels = next(iter(raw_train_data))  # 第一个批次
print("EXAMPLES: \n", examples, "\n")
print("LABELS: \n", labels)

inputs = {
    'age': tf.keras.Input([1], dtype=tf.float32),
    'fare': tf.keras.Input([1], dtype=tf.float32)
}

print(inputs)
output = tf.keras.layers.Concatenate()([input for name, input in inputs.items() if name in ['age', 'fare']])
print(output.shape)
output = tf.keras.layers.Dense(1, use_bias=True)(output)
model = tf.keras.Model(inputs=inputs, outputs=output)

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

model.fit(raw_train_data, epochs=1)
test_loss, test_accuracy = model.evaluate(raw_test_data)
print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))
model.save('tmp/model/office', save_format='tf')
