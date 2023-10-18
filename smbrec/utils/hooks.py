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

import six
import numpy as np
import tensorflow as tf
from tensorflow.python.training.basic_session_run_hooks import NeverTriggerTimer, SecondOrStepTimer

CS = "\033[1m\033[32m"
CE = "\033[0m\033[0m "


class TrainLogsHook(tf.estimator.SessionRunHook):
    """Prints the given tensors every N local steps, every N seconds, or at end.
  
    The tensors will be printed to the log, with `INFO` severity. If you are not
    seeing the logs, you might want to add the following line after your imports:
  
    ```python
      tf.logging.set_verbosity(tf.logging.INFO)
    ```
  
    Note that if `at_end` is True, `tensors` should not include any tensor
    whose evaluation produces a side effect such as consuming additional inputs.
    """

    def __init__(self, tensors, every_n_iter=None, every_n_secs=None,
                 at_end=False, formatter=None):
        """Initializes a `TrainLogsHook`.
    
        Args:
          tensors: `dict` that maps string-valued tags to tensors/tensor names,
              or `iterable` of tensors/tensor names.
          every_n_iter: `int`, print the values of `tensors` once every N local
              steps taken on the current worker.
          every_n_secs: `int` or `float`, print the values of `tensors` once every N
              seconds. Exactly one of `every_n_iter` and `every_n_secs` should be
              provided.
          at_end: `bool` specifying whether to print the values of `tensors` at the
              end of the run.
          formatter: function, takes dict of `tag`->`Tensor` and returns a string.
              If `None` uses default printing all tensors.
    
        Raises:
          ValueError: if `every_n_iter` is non-positive.
        """
        only_log_at_end = (
            at_end and (every_n_iter is None) and (every_n_secs is None))
        if (not only_log_at_end and
                (every_n_iter is None) == (every_n_secs is None)):
            raise ValueError(
                "either at_end and/or exactly one of every_n_iter and every_n_secs "
                "must be provided.")
        if every_n_iter is not None and every_n_iter <= 0:
            raise ValueError("invalid every_n_iter=%s." % every_n_iter)
        if not isinstance(tensors, dict):
            self._tag_order = tensors
            tensors = {item: item for item in tensors}
        else:
            self._tag_order = tensors.keys()
        self._tensors = tensors
        self._formatter = formatter
        self._timer = (
            NeverTriggerTimer() if only_log_at_end else
            SecondOrStepTimer(every_secs=every_n_secs, every_steps=every_n_iter))
        self._log_at_end = at_end
        self._every_n_iter = every_n_iter

    def begin(self):
        self._timer.reset()
        self._iter_count = 0
        # Convert names to tensors if given
        self._current_tensors = {tag: _as_graph_element(tensor)
                                 for (tag, tensor) in self._tensors.items()}

    def before_run(self, run_context):  # pylint: disable=unused-argument
        self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
        if self._should_trigger:
            return tf.estimator.SessionRunArgs(self._current_tensors)
        else:
            return None

    def _log_tensors(self, tensor_values):
        end = "\033[3A\r"
        if self._iter_count % self._every_n_iter == 0:
            end = ""
        train_logs = " - Training Step: " + str(self._iter_count) + "\n"
        original = np.get_printoptions()
        np.set_printoptions(suppress=True)
        elapsed_secs, _ = self._timer.update_last_triggered_step(self._iter_count)
        if self._formatter:
            tf.compat.v1.logging.info(self._formatter(tensor_values))
        else:
            for tag in self._tag_order:
                train_logs += "| " + tag + ": " + \
                              "%.4f" % tensor_values[tag] + " "
            if elapsed_secs is not None:
                train_logs += "| Elapsed Secs: " + \
                              "%.4f" % elapsed_secs + " "
            train_logs += "|\n--" + end
            tf.compat.v1.logging.info("%s", train_logs)
        np.set_printoptions(**original)

    def after_run(self, run_context, run_values):
        _ = run_context
        if self._should_trigger:
            self._log_tensors(run_values.results)

        self._iter_count += 1

    def end(self, session):
        if self._log_at_end:
            values = session.run(self._current_tensors)
            self._log_tensors(values)


def _as_graph_element(obj):
    """Retrieves Graph element."""
    graph = tf.compat.v1.get_default_graph()
    if not isinstance(obj, six.string_types):
        if not hasattr(obj, "graph") or obj.graph != graph:
            raise ValueError("Passed %s should have graph attribute that is equal "
                             "to current graph %s." % (obj, graph))
        return obj
    if ":" in obj:
        element = graph.as_graph_element(obj)
    else:
        element = graph.as_graph_element(obj + ":0")
        # Check that there is no :1 (e.g. it's single output).
        try:
            graph.as_graph_element(obj + ":1")
        except (KeyError, ValueError):
            pass
        else:
            raise ValueError("Name %s is ambiguous, "
                             "as this `Operation` has multiple outputs "
                             "(at least 2)." % obj)
    return element
