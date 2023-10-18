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

import copy
import abc
import dill
import functools
import importlib
import os
import tensorflow as tf
from datetime import datetime
from smbrec.inputs.selector import InputSelector
from tensorflow.python.estimator import util
from tensorflow.python.estimator.canned.optimizers import get_optimizer_instance


def maybe_expand_dim(tensor):
    """Expand the dim of `tensor` with static rank 1."""
    with tf.name_scope('maybe_expand_dim'):
        if isinstance(tensor, tf.SparseTensor):
            raise ValueError('SparseTensor labels are not supported.')
        static_shape = tensor.shape
        if static_shape is None:
            return tensor
        return tf.expand_dims(tensor, -1) if static_shape.ndims == 1 else tensor


class ReturnSpec(object):
    """ forward的输出

    Params:
        logits: 网络的输出
        export_predicts: predict额外的输出，predict输出包含 output、export_predicts、export_outputs
        export_outputs: serving额外的输出，serving输出包含 output、export_outputs
    """
    __slots__ = ('logits', 'export_predicts', 'export_outputs')

    def __init__(self,
                 logits,
                 export_predicts={},
                 export_outputs={}):
        self.logits = logits
        self.export_predicts = export_predicts
        self.export_outputs = export_outputs


class EstimatorModel(abc.ABC):
    """ 模型的基类，内部通过封装estimator来实现模型的训练和
    其他一系列特性。构造函数的参数是该模型的模型结构的参数，
    跟训练以及评估等相关的参数通过compile的参数传入

    """

    def __init__(self, feature_configs, model_config=None, **kwargs):
        self.feature_configs = feature_configs
        self.model_config = model_config
        self.config = kwargs.get('config', None)
        self.kwargs = kwargs
        if 'config' in self.kwargs:
            self.kwargs.pop('config')
        self.estimator = None
        self.train_config = self.config.get('train_params', None) if self.config else None
        self.model_path = self.train_config.get('model_path') \
            if self.train_config and 'model_path' in self.train_config \
            else os.path.join(os.getcwd(), 'tmp', str(int(datetime.now().timestamp())))
        checkpoint_config = self.train_config.get('checkpoint', None)
        self.checkpoint_dir = checkpoint_config.get('path') \
            if checkpoint_config and 'path' in checkpoint_config \
            else os.path.join(os.getcwd(), 'tmp', str(int(datetime.now().timestamp())))
        self.checkpoint_steps = checkpoint_config.steps \
            if checkpoint_config and 'steps' in checkpoint_config \
            else 1000
        self.keep_checkpoint_max = checkpoint_config.keep_checkpoint_max \
            if checkpoint_config and 'keep_checkpoint_max' in checkpoint_config \
            else 3
        self.targets = [feature_config.name for feature_config in self.feature_configs.values()
                        if feature_config.is_label] if self.feature_configs is not None else []
        self.mode = None
        self.online = False

    def compile(self,
                optimizer='rmsprop',
                loss=None,
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                **kwargs):
        """ 在实例化模型后，需要运行compile来构造内部的estimator对象
        """
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.loss_weight = loss_weights
        self.weighted_metrics = weighted_metrics
        self.recompile()

    def recompile(self):
        self.estimator = tf.estimator.Estimator(
            model_fn=EstimatorModel.model_fn,
            model_dir=self.checkpoint_dir,
            config=self.create_run_config(),
            params={'model': self}
        )

    @property
    def metrics_names(self):
        return [metric if isinstance(metric, str) else metric.name for metric in self.metrics]

    @abc.abstractmethod
    def forward(self, inputs=None):
        """ 模型的前向运算
        forward是Model的前向运算函数，网络结构定义在这里，采用函数式api的方式定义网络结构，而不需要
        在__init__函数中提前实例化Layer

        Args:
            inputs: 模型的输入，默认采用字典的方式

        Returns:
            outputs，模型的输出
        """
        raise NotImplementedError

    def serving_outputs(self, logits):
        """ 获取模型serving的输出

        Args:
            logits: 模型call的返回值

        Returns:
            模型serving的输出，默认等于logits
        """
        return logits

    def fit(self, input_fn, hooks=None, steps=None):
        hooks = hooks if hooks is not None else []

        self.estimator.train(input_fn=input_fn.get_input_fn() if isinstance(input_fn, InputSelector) else input_fn,
                             steps=steps,
                             hooks=hooks)
        return self

    def save(self,
             filepath,
             serving_input_receiver_fn=None,
             checkpoint_path=None,
             as_text=False):
        """ 导出模型

        Args:
            filepath: 导出模型的保存根路径，模型会导出到一个时间戳目录
            serving_input_receiver_fn: 模型输入
            checkpoint_path: checkpoint的路径，默认加载最新时间戳版本
            as_text: 是否采用text格式保存

        Returns:
            模型导出的路径
        """
        if serving_input_receiver_fn is None and self.feature_configs is None:
            raise ValueError('Both serving_input_receiver_fn and feature_configs are none.')

        if serving_input_receiver_fn is None:
            feature_spec = self.feature_configs.get_feature_spec(exclude_labels=True)
            serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

        self.online = True
        self.recompile()
        path = self.estimator.export_saved_model(
            export_dir_base=filepath,
            serving_input_receiver_fn=serving_input_receiver_fn,
            as_text=as_text,
            checkpoint_path=checkpoint_path
        )
        self.online = False
        self.recompile()
        return path.decode("utf-8")

    def latest_checkpoint(self):
        return self.estimator.latest_checkpoint()

    def evaluate(self,
                 input_fn,
                 steps=None,
                 hooks=None,
                 checkpoint_path=None,
                 name=None):
        evaluation = self.estimator.evaluate(
            input_fn=input_fn.get_input_fn() if isinstance(input_fn, InputSelector) else input_fn,
            steps=steps,
            hooks=hooks,
            checkpoint_path=checkpoint_path,
            name=name)
        return evaluation

    def predict(self,
                input_fn,
                predict_keys=None,
                hooks=None,
                checkpoint_path=None):
        for prediction in self.estimator.predict(
                input_fn=input_fn.get_input_fn() if isinstance(input_fn, InputSelector) else input_fn,
                predict_keys=predict_keys,
                hooks=hooks,
                checkpoint_path=checkpoint_path):
            yield prediction

    def create_run_config(self):
        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=self.train_config.get("inter_op_parallelism_threads", 3),
            intra_op_parallelism_threads=self.train_config.get("intra_op_parallelism_threads", 5))
        session_config.gpu_options.allow_growth = True

        config = tf.estimator.RunConfig(
            save_checkpoints_steps=self.checkpoint_steps,
            keep_checkpoint_max=self.keep_checkpoint_max,
            session_config=session_config,
            log_step_count_steps=self.train_config.get("log_step_count_steps", 100),
            #train_distribute=self.distribute_strategy,
            #eval_distribute=self.distribution_strategy  # distributed evaluation or not
        )
        return config

    def create_loss(self, inputs, logits, labels):
        """ 计算损失函数

        Args:
            inputs: 样本特征
            logits: 模型前向输出
            labels: 样本label

        Returns:
            loss
        """
        if callable(self.loss):
            _loss_fn = self.loss
        elif self.loss.find('.') == -1:
            _loss_fn = getattr(tf.nn, self.loss)
        else:
            pos = self.loss.rfind('.')
            module = self.loss[:pos]
            module = importlib.import_module('.', module)
            clazz = self.loss[pos+1:]
            _loss_fn = getattr(module, clazz)

        loss_fn_args = util.fn_args(_loss_fn)

        args = []
        # NOTE(rabin): loss函数参数必须label在logit前
        if labels is not None:
            if isinstance(labels, dict):
                for name in labels:
                    labels[name] = maybe_expand_dim(labels[name])
            else:
                labels = maybe_expand_dim(labels)
            args.append(labels)
        args.append(logits)

        # NOTE(rabin): 允许自定义损失函数，接收inputs、mode等输入
        kwargs = {}
        if 'inputs' in loss_fn_args:
            kwargs['inputs'] = inputs
        if 'mode' in loss_fn_args:
            kwargs['mode'] = self.mode

        unweighted_loss = _loss_fn(*tuple(args), **kwargs)
        losses = tf.compat.v1.losses.compute_weighted_loss(
            unweighted_loss, weights=1.0, reduction=tf.compat.v1.losses.Reduction.MEAN)
        reg_loss = tf.compat.v1.losses.get_regularization_loss()

        return losses + reg_loss

    def custom_metric_ops(self, labels, predictions):
        """ 自定义的metric ops

        Args:
            labels: 标签
            predictions: 预测值

        Returns:
            metric ops字典，{'name': metric}
        """
        return None

    def create_eval_metric_ops(self, labels, predictions):
        def _get_metric(metric):
            if isinstance(metric, str):
                metric_name = metric
                metric_fn = getattr(tf.compat.v1.metrics, metric)
            else:
                metric_fn = metric
                metric_name = metric_fn.func.__name__ if isinstance(metric_fn, functools.partial) \
                    else metric_fn.__name__
            return (metric_name, metric_fn)

        def _mean_metric(values, prefix=""):
            ops = {}
            if isinstance(values, dict):
                for name, value in values.items():
                    ops["/".join([name, prefix, 'mean'])] = tf.compat.v1.metrics.mean(value)
            else:
                ops["/".join([prefix, 'mean'])] = tf.compat.v1.metrics.mean(values)
            return ops

        metric_ops = self.custom_metric_ops(labels, predictions)
        metric_ops = {} if metric_ops is None else metric_ops
        metric_ops.update(_mean_metric(predictions, 'prediction'))
        metric_ops.update(_mean_metric(labels, 'label'))

        if not self.metrics:
            return metric_ops

        is_personalize = False if isinstance(self.metrics[0], str) else True
        if not is_personalize:  # ['metrics1', 'metrics2']
            for metric in self.metrics:
                metric_name, metric_fn = _get_metric(metric)
                if len(self.targets) == 1:
                    index = metric_fn(labels=labels, predictions=predictions)
                    metric_ops.update({metric_name: index})
                else:
                    for target in self.targets:
                        if target not in labels or (isinstance(predictions, dict) and
                                                    target not in predictions):
                            continue
                        predict = predictions[target] if isinstance(predictions, dict) \
                            else predictions
                        index = metric_fn(labels=labels[target], predictions=predict)
                        metric_ops.update({target + "'s " + metric_name: index})

        else:  # [{'label1': ['metrics1'], 'label2': ['metrics1', 'metrics2']}]
            for target, metrics in self.metrics.items():
                for metric in metrics:
                    metric_name, metric_fn = _get_metric(metric)
                    index = metric_fn(labels=labels[target], predictions=predictions[target])
                    metric_ops.update({target + "'s " + metric_name: index})
        return metric_ops

    def create_train_op(self, loss):
        """ 构建train_op

        Args:
            loss: 损失函数

        Returns:
            train_op
        """
        optimizer = get_optimizer_instance(self.optimizer, 0.01)
        train_op = optimizer.minimize(loss=loss, global_step=tf.compat.v1.train.get_global_step())
        return train_op

    @staticmethod
    def model_fn(features, labels, mode, params, config):
        """ model_fn的具体实现

        Args:
            features: 特征，dict形式
            labels: 标签，tf.Tensor
            mode: tf.estimator.ModeKeys

        Returns:
            EstimatorSpec
        """
        assert 'model' in params
        model = params['model']
        model.mode = mode
        logits = model.forward(inputs=features)
        return_spec = logits if isinstance(logits, ReturnSpec) else ReturnSpec(logits=logits)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = model.serving_outputs(logits=return_spec.logits)
            outputs = {
                tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: predictions
            } if not isinstance(predictions, dict) else predictions
            export_outputs = return_spec.export_outputs
            export_outputs.update(outputs)
            export_predicts = return_spec.export_predicts
            export_predicts.update(export_outputs)
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=export_predicts,
                export_outputs={'': tf.estimator.export.PredictOutput(export_outputs)}
            )

        eval_metric_ops = {}
        if mode == tf.estimator.ModeKeys.EVAL:
            predictions = model.serving_outputs(logits=return_spec.logits)
            eval_metric_ops = model.create_eval_metric_ops(labels, predictions)

        loss = model.create_loss(inputs=features,
                                 logits=return_spec.logits,
                                 labels=labels)

        # To support multi task learning, return a {name->loss} loss map
        total_loss = loss
        if isinstance(loss, dict):
            total_loss = sum(loss.values())

        with tf.name_scope('Losses'):
            tf.summary.scalar('total_loss', total_loss)
            if isinstance(loss, dict):
                for key, value in loss.items():
                    tf.summary.scalar(key+'_loss', value)

        train_op = model.create_train_op(loss)

        hooks = [
            tf.estimator.LoggingTensorHook(model.loggings, every_n_iter=500)
        ]
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op,
            training_hooks=hooks,
            eval_metric_ops=eval_metric_ops
        )

