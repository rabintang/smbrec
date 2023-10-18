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


# generate dummy dataset
def serialize_example(val, label):
    features = {
        'color': tf.train.Feature(bytes_list=tf.train.BytesList(value=val)),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()


tfrecord_writer = tf.io.TFRecordWriter('./color.tfrecord')
for val, label in [([b'G', b'R'], 1), ([b'B'], 1), ([b'B', b'G'], 0), ([b'R'], 1)]:
    tfrecord_writer.write(serialize_example(val, label))
tfrecord_writer.close()


# load the data generate above
def parse(example_proto):
    feature_description = {
        'color': tf.io.VarLenFeature(tf.string),  # ** VarLenFeature **
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    labels = parsed_features.pop('label')
    return parsed_features, labels


dataset = tf.data.TFRecordDataset('./color.tfrecord').map(parse).repeat(5).batch(2)

# feature column & inputs.
color_cat = tf.feature_column.categorical_column_with_vocabulary_list(
    key='color', vocabulary_list=["R", "G", "B"])

color_emb = tf.feature_column.embedding_column(color_cat, dimension=4, combiner='mean')

inputs = {
    'color': tf.keras.layers.Input(name='color', shape=(None,), sparse=True, dtype=tf.string)
}

# build model
deep = tf.keras.layers.DenseFeatures([color_emb, ])(inputs)
output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(deep)
model = tf.keras.Model(inputs, output)
model.compile(optimizer='adam', loss='binary_crossentropy')

model.summary()

model.fit(dataset, epochs=5)

@tf.function
def signature_function(inputs):
    """ 模型导出 savedModel
    https://stackoverflow.com/questions/59142040/tensorflow-2-0-how-to-change-the-output-signature-while-using-tf-saved-model

    Args:
        inputs:

    Returns:

    """
    predictions = model(inputs)
    outputs = {
        'predictions': predictions
    } if not isinstance(predictions, dict) else predictions
    return outputs


inputs = {name: value.type_spec if isinstance(value.type_spec, tf.TensorSpec) else \
        NamedSparseTensorSpec(shape=value.shape, dtype=value.dtype, name=name)
              for name, value in model.input.items()}
model.save('./dummy_model', save_format='tf', signatures=signature_function.get_concrete_function(
                   inputs=inputs
               ))
