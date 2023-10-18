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

import abc
import functools
import logging
import logging.handlers
import threading
import numpy as np
import random
import os
import sys
import tensorflow as tf
import time

from smbrec.core.config import FeatureConfigs, SampleConfig
from smbrec.utils import hdfs_utils, file_utils
from smbrec.utils.commandline_config import Config


class BaseLauncher(object):
    def __init__(self, model, config, feature_configs):
        """ pipeline执行器基类

        Args:
            model: 模型实例
            config: pipeline config
            feature_configs: 特征集config
        """
        self.model = model
        self.config = config
        self.feature_configs = feature_configs
        self.train_config = self.config.get(
            'train_params', Config({}, name="dict train_config", read_command_line=False))
        self.export_config = self.config.get(
            'export_params', Config({}, name="dict export_config", read_command_line=False))
        self.eval_config = self.config.get(
            'eval_params', Config({}, name="dict eval_config", read_command_line=False))
        self.status = True  # 运行状态，True为正常，False为异常
        self.session = None
        tmp_dir = os.environ.get("SMBREC_TMP_DIR", "/tmp/smbrec/") + self.config.version
        train_data_config = self.train_config.data
        assert "path" in train_data_config, "Cannot find train data path."
        # 训练数据默认shuffle，且丢弃最后一个不完整batch
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True
        if 'drop_remainder' not in train_data_config:
            train_data_config['drop_remainder'] = True
        self.train_sample_config = SampleConfig.from_dict(train_data_config)
        self.epochs = self.train_sample_config.epochs
        self.train_sample_config.epochs = 1
        self.checkpoint_config = self.train_config.get('checkpoint', {})
        self.checkpoint_dir = self.checkpoint_config.get('path', tmp_dir + '/checkpoints/')
        self.is_streaming = self.train_sample_config.mode == 'streaming'
        timestamp = str(int(time.time()))
        if not self.is_streaming:
            self.checkpoint_dir = os.path.join(self.checkpoint_dir, timestamp)
        self.tensorboard_dir = os.path.join(self.train_config.get('tensorboard_dir',
                                                                  os.path.join(tmp_dir, 'tensorboard')),
                                            timestamp)
        self.model_saved_dir = os.path.join(tmp_dir, 'saved_models')  # 模型导出本地临时保存路径
        self.latest_checkpoint = None  # 上一次导出模型的checkpoint版本
        self.timestamp_since_last_saving = int(time.time())  # 上一次导出模型的时间
        self.checkpoint_fail_times = 0  # checkpoint失败数
        self.max_checkpoint_fail_times = 3  # checkpoint最大失败数
        self.set_environ()
        self.set_random_seed()
        self.set_devices()
        self.set_logger(self.checkpoint_dir)

    def set_environ(self):
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['KMP_BLOCKTIME'] = '1'
        os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
        os.environ['KMP_SETTINGS'] = 'TRUE'

    def set_random_seed(self):
        random_seed = self.train_config.get('random_seed', None)
        if random_seed is not None:
            os.environ['PYTHONHASHSEED'] = str(random_seed)
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            np.random.seed(random_seed)
            random.seed(random_seed)
            try:
                tf.set_random_seed(random_seed)
            except:
                tf.random.set_seed(random_seed)

    def set_logger(self, log_dir, log_file_name='tensorflow.log', log_level=tf.compat.v1.logging.INFO):
        tf.compat.v1.logging.set_verbosity(log_level)
        logger = logging.getLogger('tensorflow')
        formatter = logging.Formatter('%(asctime)s %(name)s: %(levelname)s: %(message)s')

        # create file handler
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        handler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(log_dir, log_file_name),
            encoding='utf8', )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

        def _exception_hook(exctype, value, traceback):
            sys.__excepthook__(exctype, value, traceback)
            tf.compat.v1.logging.info('Exception captured:', exc_info=(exctype, value, traceback))

        sys.excepthook = _exception_hook

    def set_devices(self):
        gpu_indices = [int(index.strip()) for index in self.train_config.get('gpus', '').split(',') if index]
        # TODO(rabin): 1.14使用接口设置visible_devices报错
        if tf.__version__ <= '1.15.0':
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(index) for index in gpu_indices])
            config = tf.compat.v1.ConfigProto(
                gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)
            )
            self.session = tf.compat.v1.Session(config=config)
            tf.compat.v1.keras.backend.set_session(self.session)
        else:
            try:
                gpus = tf.config.list_physical_devices(device_type='GPU')
            except:
                gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

            visible_gpus = [gpus[index] for index in gpu_indices] if gpu_indices else gpus
            try:
                tf.config.set_visible_devices(devices=visible_gpus, device_type='GPU')
            except:
                tf.config.experimental.set_visible_devices(devices=visible_gpus, device_type='GPU')

            """for gpu in visible_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)"""

    @abc.abstractmethod
    def get_inputs(self):
        """ 获取pipeline的输入

        Returns:
            (train_input, eval_input)
        """
        raise NotImplementedError

    def train(self, data):
        """ 模型训练

        Args:
            data: 训练数据集

        Returns:
            None
        """
        # 1. 增量训练模式，需要加载历史最新版本的模型
        if self.train_config.mode == 'online' and file_utils.exists(self.train_config.model_path):
            # download latest checkpoint
            assert self.train_config.model_path, "training mode is online, so must set model path."
            latest = sorted(file_utils.listdir(self.train_config.model_path), reverse=True)
            if latest:
                latest = latest[0]
                print("download latest checkpoint from " + latest)
                file_utils.copy(latest, self.checkpoint_dir)

        # 2. 训练模型
        self._train_model(data)

    @abc.abstractmethod
    def _train_model(self, data):
        """ 执行模型fit训练接口

        Args:
            data: 训练数据集

        Returns:
            None
        """
        raise NotImplementedError

    def _upload_checkpoint(self, filepath=None):
        """ 上传checkpoint
        上传checkpoint到model path

        Args:
            filepath: 待上传的checkpoint文件路径
        """
        if not self.train_config.model_path:
            return

        if filepath is None:
            if not self.is_streaming:
                filepath = self.checkpoint_dir
            else:
                filenames = sorted(os.listdir(self.checkpoint_dir))\
                    if os.path.exists(self.checkpoint_dir) else []
                if filenames:
                    filepath = os.path.join(self.checkpoint_dir, filenames[-1])
        if filepath:
            filename = os.path.basename(filepath)
            dest_path = os.path.join(self.train_config.model_path, filename)
            print("upload checkpoint from " + filepath + " to " + dest_path)
            file_utils.copy(filepath, dest_path)

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
        print("evaluation results: ", results)

    def predict(self, data, path, model=None):
        """ 输出模型预测结果

        Args:
            data: 预测输出数据集
            path: 输出结果路径
            model: 待评估的模型

        Returns:
            None
        """
        model = model if model else self.model
        with open(path, 'w') as ouf:
            for idx, prediction in enumerate(model.predict(data)):
                if idx == 0:
                    ouf.write('|'.join(prediction.keys()) + "\n")
                if idx % 10000 == 0 and idx != 0:
                    print('finish: ', idx)
                ouf.write('|'.join([','.join([str(v) for v in val]) if type(val) in (list, tuple, np.ndarray) \
                    else str(val) for val in prediction.values()]) + "\n")
            print('total: ', idx)

    def export(self, checkpoint_path=None):
        """ 导出模型到目标目录

        Args:
            checkpoint_path: 待导出的模型checkpoint地址

        Returns:
            None
        """
        if not self.export_config.path:
            return

        saved_path = self._export_model(self.model_saved_dir, checkpoint_path)
        if self.export_config.get('compress', False):
            file_utils.compress(saved_path, saved_path)
        # sync saved model
        basename = os.path.basename(saved_path)
        if not file_utils.exists(self.export_config.path):
            file_utils.create_dir(self.export_config.path)
        dest_path = os.path.join(self.export_config.path, basename)
        print("upload saved model from " + saved_path + " to " + dest_path)
        file_utils.copy(saved_path, dest_path)

    @abc.abstractmethod
    def _export_model(self, save_dir, checkpoint_path=None):
        """ 导出模型到指定目录

        Args:
            save_dir: 模型保存的根目录
            checkpoint_path: 导出指定的checkpoint路径

        Returns:
            模型保存的目标目录
        """
        raise NotImplementedError

    def export_continuous(self):
        """ 增量模型评估与导出
        streaming模式下，定时导出模型
        """
        if self.export_config.get('mode', 'batch') != 'streaming' or \
                self.export_config.get('path', None) is None:
            return

        streaming_params = self.export_config.get('streaming_params', {})
        self.max_checkpoint_fail_times = streaming_params.get('max_fail_times', self.max_checkpoint_fail_times)
        export_seconds = streaming_params.get('seconds', 10 * 60)  # 默认每10mins导出模型
        while True and self.checkpoint_fail_times <= self.max_checkpoint_fail_times:
            try:
                elapse_seconds = int(time.time()) - self.timestamp_since_last_saving
                filenames = sorted(os.listdir(self.checkpoint_dir)) if os.path.exists(self.checkpoint_dir) else []
                # 时间未达到，或者checkpoint未更新
                if elapse_seconds < export_seconds or \
                        len(filenames) == 0 or \
                        filenames[-1] == self.latest_checkpoint:
                    time.sleep(10)
                    continue

                # 加载checkpoint导出模型
                self.export(os.path.join(self.checkpoint_dir, filenames[-1]))
                self.latest_checkpoint = filenames[-1]
                self.timestamp_since_last_saving = int(time.time())
                self._upload_checkpoint(self.latest_checkpoint)
                time.sleep(10)
            except Exception as e:
                print('checkpoint model fail, msg:', str(e))
                self.checkpoint_fail_times += 1
        self.status = False

    def run(self):
        """ 执行pipeline config

        Returns:
            None
        """
        (train_input, eval_input) = self.get_inputs()
        # TODO(rabin): 1.14 版本下暂时不支持增量导出模型
        if tf.__version__ < '1.15.0':
            self.train(train_input)
            if not self.status:
                raise RuntimeError('Train fail.')
        else:
            train_thread = threading.Thread(target=functools.partial(self.train, data=train_input))
            train_thread.start()
            checkpoint_thread = threading.Thread(target=self.export_continuous)
            checkpoint_thread.start()
            train_thread.join()
            if not self.status:
                raise RuntimeError('Train fail.')
            checkpoint_thread.join()
            if self.checkpoint_fail_times > self.max_checkpoint_fail_times:
                raise RuntimeError('Checkpoint fail.')

        self._upload_checkpoint()

        if eval_input:
            self.evaluate(eval_input)
        if self.eval_config.get('need_predictions', False):
            assert 'predictions_path' in self.eval_config
            self.predict(eval_input, self.eval_config.predictions_path)

        if self.export_config.get('path', None):
            self.export()
