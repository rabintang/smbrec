# -*- coding: UTF-8 -*-
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

import traceback
import smbrec
import os
import sys
import tensorflow as tf
import time

from base_launcher import BaseLauncher
from smbrec.core.config import FeatureConfigs, SampleConfig
from smbrec.inputs import InputSelector


if tf.__version__ >= '2.0':
      tf = tf.compat.v1


class KerasLauncher(BaseLauncher):
    def __init__(self, model, config, feature_configs):
        super(KerasLauncher, self).__init__(model=model, config=config, feature_configs=feature_configs)
        # NOTO(rabin): 1.14需要tables_initializer，否则VocabEncoder将报错
        if tf.__version__ < '1.15.0':
            tf.compat.v1.tables_initializer().run(session=self.session)

    def get_inputs(self):
        # load data
        train_input_selector = InputSelector(sample_config=self.train_sample_config,
                                             feature_configs=self.feature_configs)
        eval_input = None
        if 'data' in self.eval_config.data and 'path' in self.eval_config.data:
            eval_sample_config = SampleConfig.from_dict(self.eval_config.data)
            eval_input_selector = InputSelector(sample_config=eval_sample_config, feature_configs=self.feature_configs)
            eval_input = eval_input_selector.get_input()

        return train_input_selector.get_input(), eval_input

    def evaluate(self, data, model=None):
        """ 评估模型离线指标

        Args:
            data: 评估数据集
            model: 待评估的模型

        Returns:
            None
        """
        model = model if model else self.model
        results = model.evaluate(data)
        results = dict(zip(model.metrics_names, results))
        print("evaluation results: ", results)

    def _train_model(self, data):
        if self.is_streaming:
            checkpoint_path = os.path.join(self.checkpoint_dir, '{timestamp}', 'cp.ckpt')
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'cp.ckpt')
        # 1. 添加callbacks
        callbacks = []
        # TODO(rabin): 1.14对checkpoint和histogram支持不友好，先去掉
        if tf.__version__ >= '1.15.0':
            cp_callback = smbrec.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_weights_only=True,
                load_weights_on_restart=self.train_config.get('mode') == 'online',
                save_freq=self.checkpoint_config.get('steps', 'epoch'),
                save_secs_freq=self.checkpoint_config.get('seconds', None),
                change_path_auto=True if self.is_streaming else False,
                verbose=1)
            callbacks.append(cp_callback)
            tb_callback = tf.keras.callbacks.TensorBoard(log_dir=self.tensorboard_dir,
                                                         histogram_freq=1,
                                                         update_freq=10,
                                                         profile_batch=10)
            callbacks.append(tb_callback)
        # 自定义callbacks
        if 'callbacks' in self.train_config:
            for callback in self.train_config.callbacks:
                pos = callback.rfind('.')
                cb_module = callback[:pos]
                cb_obj = callback[pos + 1:]
                module = cb_module
                __import__(module)
                cb = getattr(sys.modules[module], cb_obj)
                callbacks.append(cb)
        if self.is_streaming:
            callbacks.append(smbrec.callbacks.StreamTaskStatus(task=self))

        # 2. 模型训练
        try:
            self.model.fit(data,
                           epochs=self.epochs,
                           callbacks=callbacks)
        except Exception as e:
            print("Train fail, error msg: ", str(e))
            traceback.print_exc()
            self._status = False

    def _export_model(self, saved_dir, checkpoint_path=None):
        saved_path = os.path.join(saved_dir, str(int(time.time())))
        self.model.save(saved_path, save_format='tf', include_optimizer=False)
        return saved_path
