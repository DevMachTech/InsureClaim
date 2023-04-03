
# Define imports
from kerastuner.engine import base_tuner 
import kerastuner as kt
from numpy import dtype
from sympy import factor 
from tensorflow import keras  

from typing import NamedTuple, Dict, Text, Any 
from tfx.components.trainer.fn_args_utils import FnArgs, DataAccessor
import tensorflow as tf 
import tensorflow_transform as tft
import cover_constants as constants


# _DENSE_FLOAT_FEATURE_KEYS = constants.DENSE_FLOAT_FEATURE_KEYS
#_VOCAB_FEATURE_KEYS = constants.VOCAB_FEATURE_KEYS
_VOCAB_FEATURE_DICT = constants.VOCAB_FEATURE_DICT
_NUMERIC_FEATURE_KEYS = constants.NUMERIC_FEATURE_KEYS
_SCALE_Z_FEATURE_KEYS = constants.SCALE_Z_FEATURE_KEYS
_VOCAB_SIZE = constants.VOCAB_SIZE
_OOV_SIZE = constants.OOV_SIZE
#_HASH_STRING_FEATURE_KEYS = constants.HASH_STRING_FEATURE_KEYS
_LABEL_KEY = constants.LABEL_KEY
_transformed_name = constants.transformed_name


_NUM_OOV_BUCKETS = constants.NUM_OOV_BUCKETS

# LABEL_KEY = 'label'

def transformed_name(key):
    key = key.replace('-', '_')
    return key + '_xf'

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


def _gzip_reader_fn(filenames):

    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')
def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example and applies TFT."""

  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
    transformed_features = model.tft_layer(parsed_features)
    return model(transformed_features)

  return serve_tf_examples_fn


def _input_fn(file_pattern,tf_transform_output,num_epochs=None,batch_size=128) -> tf.data.Dataset:

    transformed_feature_spec =(
        tf_transform_output.transformed_feature_spec().copy()
    )

    # create batches of features and labels
    dataset  = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader = _gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=_LABEL_KEY
    )

    return dataset

def model_builder(hp):

    num_hidden_layers = hp.get('hidden_layers')
    # Get the learning rate from the Tuner results
    hp_learning_rate = hp.get('learning_rate')
    # hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    input_numeric = [
        tf.keras.layers.Input(name=transformed_name(colname), shape=(1,), dtype=tf.float32) for colname in _NUMERIC_FEATURE_KEYS
    ]

    input_scaler = [
       tf.keras.layers.Input(name=transformed_name(colname), shape=(1,), dtype=tf.float32) for colname in _SCALE_Z_FEATURE_KEYS 
    ]
    input_categorical = [
        tf.keras.layers.Input(name=transformed_name(colname), shape=(vocab_size + _NUM_OOV_BUCKETS,), dtype=tf.float32) for colname, vocab_size in _VOCAB_FEATURE_DICT.items()
    ]

    input_layers = input_numeric + input_categorical + input_scaler


    input_numeric = tf.keras.layers.concatenate(input_numeric)
    input_categorical = tf.keras.layers.concatenate(input_categorical)
    input_scaler = tf.keras.layers.concatenate(input_scaler)

    #deep = input_categorical

    deep = tf.keras.layers.concatenate([input_numeric, input_categorical, input_scaler])

    for i in range(num_hidden_layers):
        # Get the number of units from the Tuner results
        num_nodes = hp.get('unit'+str(i))
        deep = tf.keras.layers.Dense(num_nodes,activation='relu')(deep)

    output = tf.keras.layers.Dense(1, activation='sigmoid')(deep)

    

    model = tf.keras.Model(input_layers, output)

    

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate = hp_learning_rate),
        metrics = 'binary_accuracy'
    )

    model.summary()

    return model

def run_fn(fn_args: FnArgs) -> None:
  """Defines and trains the model.
  Args:
    fn_args: Holds args as name/value pairs. Refer here for the complete attributes: 
    https://www.tensorflow.org/tfx/api_docs/python/tfx/components/trainer/fn_args_utils/FnArgs#attributes
  """

  # Callback for TensorBoard
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=fn_args.model_run_dir, update_freq='batch')
  
  # Load transform output
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
  
  # Create batches of data good for 10 epochs
  train_set = _input_fn(fn_args.train_files, tf_transform_output, 10)
  val_set = _input_fn(fn_args.eval_files, tf_transform_output, 10)

  # Load best hyperparameters
  hp = fn_args.hyperparameters.get('values')

  # Build the model
  model = model_builder(hp)

  signatures = {
    'serving_default':
     _get_serve_tf_examples_fn(model,
     tf_transform_output).get_concrete_function(
       tf.TensorSpec(
         shape=[None],
         dtype=tf.string,
         name='examples')),
         }

  # Train the model
  model.fit(
      x=train_set,
      validation_data=val_set,
      callbacks=[tensorboard_callback]
      )
  
  # Save the model
  # signatures = {'serving_default': 'Building Dimension_xf'}
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
