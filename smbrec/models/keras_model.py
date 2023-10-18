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
import os
import tensorflow as tf
import smbrec.metrics
from smbrec.core.config import FeatureFormat, DATATYPE_2_TFTYPE, SPARSE_FEATURE_FORMAT
from smbrec.utils.commandline_config import Config


class KerasModel(tf.keras.Model):
    """ 模型的基类，建议模型都继承自该基类
    smbrec.models.Model继承自tf.keras.Model，但是相比于一般的子类化模型实现，smbrec.models.Model
    实际是使用的函数式api的接口。因此不需要在Model.__init__中定义需要使用的Layer，可以像函数式api一样，
    在 forward 函数中实现 outputs 的计算即可。

    functional api vs model subclassing:
    https://www.tensorflow.org/guide/keras/functional
    https://blog.tensorflow.org/2019/01/what-are-symbolic-and-imperative-apis.html

    Args:
        feature_configs: 特征的配置字典
        model_config: 模型参数，即模型配置文件中的model_config的参数
        kwargs: tf.keras.Model的除掉inputs和outputs外的其他参数
    """
    def __init__(self, feature_configs, model_config=None, **kwargs):
        super(KerasModel, self).__init__()
        self.feature_configs = feature_configs
        self.model_config = model_config
        self.config = kwargs.get('config', None)
        self.kwargs = kwargs
        if 'config' in self.kwargs:
            self.kwargs.pop('config')

    def compile(self,
                optimizer='rmsprop',
                loss=None,
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                **kwargs):
        # Replicates Model(inputs=[inputs], outputs=[outputs]) of functional model.
        inputs = self.get_inputs()
        outputs = self.forward(inputs)
        super(KerasModel, self).__init__(inputs=inputs, outputs=outputs, **self.kwargs)

        super(KerasModel, self).compile(optimizer=optimizer,
                                        loss=loss,
                                        metrics=metrics,
                                        loss_weights=loss_weights,
                                        weighted_metrics=weighted_metrics,
                                        run_eagerly=run_eagerly,
                                        **kwargs)

    def get_inputs(self):
        inputs_dict = dict()

        for name, feature_config in self.feature_configs.items():
            if feature_config.is_label:
                continue
            shape = [1]
            if feature_config.format == FeatureFormat.VECTOR:
                shape = [feature_config.dimension]
            elif feature_config.format == FeatureFormat.LIST:
                shape = [None]
            elif feature_config.format == FeatureFormat.KV:
                shape = [feature_config.vocab_size]
            is_sparse = True if feature_config.format in SPARSE_FEATURE_FORMAT else False
            inputs_dict[name] = tf.keras.layers.Input(shape=shape, name=name,
                                                      dtype=DATATYPE_2_TFTYPE[feature_config.data_type],
                                                      sparse=is_sparse)

        return inputs_dict

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

    @tf.function
    def signature_function(self, inputs):
        """ 模型导出 savedModel
        https://stackoverflow.com/questions/59142040/tensorflow-2-0-how-to-change-the-output-signature-while-using-tf-saved-model

        Args:
            inputs:

        Returns:

        """
        predictions = self.serving_outputs(self(inputs))
        outputs = {
            'predictions': predictions
        } if not isinstance(predictions, dict) else predictions
        return outputs

    def save(self,
             filepath,
             overwrite=True,
             include_optimizer=True,
             save_format=None,
             options=None,
             save_traces=True):
        """ 模型保存
        对tf格式保存进行了改写，以支持导出模型时自动生成signatures

        Args:
            filepath: 模型保存路径
            overwrite: 是否覆盖旧模型
            include_optimizer: 是否导出optimizer
            save_format: 保存格式，支持tf和h5
            options: tf保存时的参数
            save_traces: tf保存时的traces
        """
        if tf.__version__ >= "2.0.0":
            from smbrec.framework.tensor_spec import NamedSparseTensorSpec
            signatures = None
            # https://zhuanlan.zhihu.com/p/195750736
            # NOTE(rabin) 对于SparseTensorSpec，默认没有指定name，因此重载了NamedSparseTensorSpec
            if save_format == 'tf':
                if tf.__version__ >= "2.4.0":
                    inputs = {name: value.type_spec if isinstance(value.type_spec, tf.TensorSpec) else \
                        NamedSparseTensorSpec(shape=value.shape, dtype=value.dtype, name=name)
                              for name, value in self.input.items()}
                else:
                    inputs = {name: value if isinstance(value, tf.TensorSpec) else \
                        NamedSparseTensorSpec(shape=value.shape, dtype=value.dtype, name=name)
                              for name, value in self._saved_model_inputs_spec.items()}
                signatures = self.signature_function.get_concrete_function(inputs=inputs)
    
            super(KerasModel, self).save(filepath=filepath,
                                         overwrite=overwrite,
                                         save_format=save_format,
                                         signatures=signatures,
                                         options=options,
                                         include_optimizer=include_optimizer)
        else:
            # https://www.tensorflow.org/api_docs/python/tf/compat/v1/keras/experimental/export_saved_model
            inputs = {t.name.strip(":0"): t for t in self.inputs}
            predictions = self.serving_outputs(self.output)
            outputs = {
                'predictions': predictions
                } if not isinstance(predictions, dict) else predictions
            signature = tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(inputs=inputs, outputs=outputs)
            export_path = os.path.join(tf.compat.as_bytes(filepath))
            builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_path)
            legacy_init_op = tf.group(tf.compat.v1.tables_initializer(), name='init_all_tables2')
            builder.add_meta_graph_and_variables(
                sess=tf.compat.v1.keras.backend.get_session(),
                tags=[tf.saved_model.SERVING],
                signature_def_map={
                    'serving_default': signature,
                },
                strip_default_attrs=True,
                legacy_init_op=legacy_init_op,
                clear_devices=True
            )
            builder.save()

    def predict_step(self, data):
        return self.serving_outputs(super(KerasModel, self).predict_step(data=data))

    def serving_outputs(self, logits):
        """ 获取模型serving的输出

        Args:
            logits: 模型call的返回值

        Returns:
            模型serving的输出，默认等于logits
        """
        return logits
    
    def get_config(self):
        config = super(KerasModel, self).get_config()
        config.update({
            'feature_configs': self.feature_configs,
            'model_config': self.model_config,
            'config': self.config,
            'kwargs': self.kwargs
        })
        return config
