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
import tempfile
import time
from tensorflow.python.keras import backend as K
from smbrec.utils import sugar_utils

if sugar_utils.compared_version("2.1.0", tf.__version__) <= 0 and sugar_utils.compared_version("2.5.0", tf.__version__) > 0:
    from tensorflow.python.distribute import distributed_file_utils
elif sugar_utils.compared_version("2.5.0", tf.__version__) <= 0:
    from tensorflow.python.keras.distribute import distributed_file_utils
else:
    from tensorflow.python.distribute import distribute_coordinator_context as dc_context


class ModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """ 模型checkpoint

    Args:
        save_secs_freq: checkpoint的时间频率，即每个多长时间checkpoint，单位秒
        change_path_auto: 是否每次checkpoint自动更新filepath，默认False
    """

    def __init__(self,
                 filepath,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto',
                 save_freq='epoch',
                 save_secs_freq=None,
                 change_path_auto=False,
                 options=None,
                 **kwargs):
        super(ModelCheckpoint, self).__init__(filepath=filepath,
                                              monitor=monitor,
                                              verbose=verbose,
                                              save_best_only=save_best_only,
                                              save_weights_only=save_weights_only,
                                              mode=mode,
                                              save_freq=save_freq,
                                              options=options,
                                              **kwargs)
        self._save_secs_freq = save_secs_freq
        self._timestamp_since_last_saving = int(time.time())
        self._change_path_auto = change_path_auto

    def on_train_begin(self, logs=None):
        super(ModelCheckpoint, self).on_train_begin()
        self._timestamp_since_last_saving = int(time.time())

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        # pylint: disable=protected-access
        self._save_model(epoch=epoch, logs=logs)

    def _should_save_on_batch(self, batch):
        """Handles batch-level saving logic, supports steps_per_execution."""
        if self.save_freq == 'epoch':
            return False

        if batch <= self._last_batch_seen:  # New epoch.
            add_batches = batch + 1  # batches are zero-indexed.
        else:
            add_batches = batch - self._last_batch_seen
        self._batches_seen_since_last_saving += add_batches
        self._last_batch_seen = batch

        time_elapsed = int(time.time()) - self._timestamp_since_last_saving
        if self._batches_seen_since_last_saving >= self.save_freq \
                and (not self._save_secs_freq or time_elapsed >= self._save_secs_freq):
            self._batches_seen_since_last_saving = 0
            self._timestamp_since_last_saving = int(time.time())
            return True
        return False

    def _get_file_path(self, epoch, logs):
        """Returns the file path for checkpoint."""
        # pylint: disable=protected-access
        try:
            # `filepath` may contain placeholders such as `{epoch:02d}` and
            # `{mape:.2f}`. A mismatch between logged metrics and the path's
            # placeholders can cause formatting to fail.
            file_path = self.filepath.format(timestamp=str(int(time.time())), epoch=epoch + 1, **logs)
        except KeyError as e:
            raise KeyError('Failed to format this callback filepath: "{}". '
                           'Reason: {}'.format(self.filepath, e))
        self._write_filepath = distributed_file_utils.write_filepath(
            file_path, self.model.distribute_strategy)
        return self._write_filepath

    def _get_file_handle_and_path(self, epoch, logs):
        """Returns the file handle and path."""
        # TODO(rchao): Replace dc_context reference with
        # distributed_training_utils.should_current_worker_checkpoint() once
        # distributed_training_utils.py no longer depends on callbacks.py.
        if not K.in_multi_worker_mode() or dc_context.get_current_worker_context(
        ).should_checkpoint:
            return None, self.filepath.format(timestamp=str(int(time.time())), epoch=epoch + 1, **logs)
        else:
            # If this is multi-worker training, and this worker should not
            # save checkpoint, we replace the filepath with a dummy filepath so
            # it writes to a file that will be removed at the end of _save_model()
            # call. This is because the SyncOnReadVariable needs to be synced across
            # all the workers in order to be read, and all workers need to initiate
            # that.
            file_handle, temp_file_name = tempfile.mkstemp()
            extension = os.path.splitext(self.filepath)[1]
            return file_handle, temp_file_name + '.' + extension


class StreamTaskStatus(tf.keras.callbacks.Callback):
    """ 任务状态检查
    根据任务状态判断是否继续训练

    Args:
        task: task实例，需要获取status字段
    """

    def __init__(self, task, **kwargs):
        super(StreamTaskStatus, self).__init__()
        self._task = task
        assert hasattr(self._task, 'status'), 'Task object must have status property.'

    def on_train_begin(self, logs=None):
        if not self._task.status:
            raise RuntimeError('task is failed.')
