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

import os
import queue
import threading
import time

import tensorflow as tf
from datetime import datetime, timedelta


class StreamingGenerator(object):
    """ 样本数据集的访问接口
    给定样本和特征配置，获取对应的 tf.data.Dataset 对象

    Args:
        sample_cfg: 样本配置
    """

    def __init__(self, sample_cfg):
        self.sample_cfg = sample_cfg
        self._streaming_params = self.sample_cfg.streaming_params
        self.file_queue = queue.Queue(self._streaming_params.buffer_size)
        self.consumer_queue = queue.Queue(self._streaming_params.timeout * 2)
        self._start_point = self._restore()
        self._lock = threading.Lock()
        self._max_checkpoint_t = None  # 最大的checkpoint时间戳
        self._worker = threading.Thread(target=self._consume)
        self._worker.start()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        t, filepath = self.file_queue.get()
        # TODO 一个时间分片可能存在多个文件，其余文件不一定参与训练
        self._checkpoint(t)
        return filepath

    def _restore(self):
        """ 恢复启动时间
        如果存在checkpoint文件，则读取checkpoint文件时间，否则采用配置文件的时间

        Returns:
            启动时间
        """
        start_point = self._streaming_params.start_point
        if os.path.exists(self._streaming_params.checkpoint):
            with open(os.path.join(self._streaming_params.checkpoint)) as f:
                data = f.read().strip()
                if data != "":
                    start_point = data
        return datetime.strptime(start_point, self._streaming_params.time_fmt)

    def _checkpoint(self, t):
        """ 写入中断恢复信息

        Args:
            t: 时间
        """
        if self._streaming_params.checkpoint is not None and \
                (self._max_checkpoint_t is None or t > self._max_checkpoint_t):
            basedir = os.path.dirname(self._streaming_params.checkpoint)
            self._lock.acquire()
            if not os.path.exists(basedir):
                os.makedirs(basedir)
            with open(self._streaming_params.checkpoint, 'w') as f:
                f.write(t.strftime(self._streaming_params.time_fmt))
            self._max_checkpoint_t = t
            self._lock.release()

    def _time2dir(self, t):
        """ 根据时间获取数据目录

        Args:
            t: 数据时间

        Returns:
            数据目录
        """
        return os.path.join(self.sample_cfg.path, "%d%02d%02d" % (t.year, t.month, t.day),
                            "%d" % t.hour, "%d" % t.minute)

    def _check_ready(self, dir):
        """ 检查目录是否就绪

        Args:
            dir: 待检查的数据目录

        Returns:
            True or False
        """
        if self._streaming_params.ready_flag:
            return tf.io.gfile.exists(os.path.join(dir, self._streaming_params.ready_flag))
        return tf.io.gfile.exists(dir)

    def _consume(self):
        """ 消费者任务
        消费分钟信息，获取对应分钟目录下的数据文件写入文件队列
        """
        while True:
            t = self.consumer_queue.get()
            dir = self._time2dir(t)
            if self._check_ready(dir):
                [
                    self.file_queue.put((t, os.path.join(dir, x)))
                    for x in sorted(tf.io.gfile.listdir(dir))
                    if x != "_SUCCESS"
                ]
            else:
                # 小于超时时间则重新放进消费队列待下次消费
                # TODO(rabin) 导致超时的样本放到队列末尾，训练乱序？
                if datetime.now() - t < timedelta(minutes=self._streaming_params.timeout):
                    self.consumer_queue.put(t)
                    time.sleep(5)

    def produce(self):
        """ 生产者任务
        生产分钟信息写入消费队列
        """
        cur = self._start_point
        while True:
            if cur < datetime.now():
                self.consumer_queue.put(cur)
                print('cursor:', datetime.strftime(cur, self._streaming_params.time_fmt),
                      ', now:', datetime.strftime(datetime.now(), self._streaming_params.time_fmt),
                      ', file queue size:', self.file_queue.qsize())
                cur = cur + timedelta(minutes=1)
            else:
                time.sleep(10)

    def start(self):
        threading.Thread(target=self.produce).start()
