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

from base_launcher import BaseLauncher
from smbrec.core.config import FeatureConfigs, SampleConfig
from smbrec.inputs import InputSelector
from smbrec.utils.hooks import TrainLogsHook


class EstimatorLauncher(BaseLauncher):
    def __init__(self, model, config, feature_configs):
        super(EstimatorLauncher, self).__init__(model=model, config=config, feature_configs=feature_configs)
        self.model.checkpoint_dir = self.checkpoint_dir
        self.model.recompile()

    def get_inputs(self):
        # load data
        train_input_selector = InputSelector(sample_config=self.train_sample_config,
                                             feature_configs=self.feature_configs)
        eval_input = None
        if 'data' in self.eval_config and 'path' in self.eval_config.data:
            eval_sample_config = SampleConfig.from_dict(self.eval_config.data)
            eval_input_selector = InputSelector(sample_config=eval_sample_config, feature_configs=self.feature_configs)
            eval_input = eval_input_selector.get_input_fn()

        return train_input_selector.get_input_fn(), eval_input

    def _train_model(self, data):
        try:
            self.model.fit(data)
        except Exception as e:
            print("Train fail, error msg: ", str(e))
            traceback.print_exc()
            self._status = False

    def _export_model(self, saved_dir, checkpoint_path=None):
        saved_path = self.model.save(saved_dir, checkpoint_path=checkpoint_path)
        return saved_path
