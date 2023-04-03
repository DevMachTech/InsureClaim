
import tensorflow as tf
import tensorflow_transform as tft

import cover_constants

# _SCALE_MINMAX_FEATURE_KEYS = cover_constants.SCALE_MINMAX_FEATURE_KEYS
#_SCALE_01_FEATURE_KEYS = cover_constants.SCALE_01_FEATURE_KEYS
_VOCAB_FEATURE_DICT = cover_constants.VOCAB_FEATURE_DICT
_NUMERIC_FEATURE_KEYS = cover_constants.NUMERIC_FEATURE_KEYS
_SCALE_Z_FEATURE_KEYS = cover_constants.SCALE_Z_FEATURE_KEYS
# _VOCAB_FEATURE_KEYS = cover_constants.VOCAB_FEATURE_KEYS
_HASH_STRING_FEATURE_KEYS = cover_constants.HASH_STRING_FEATURE_KEYS
_LABEL_KEY = cover_constants.LABEL_KEY
_NUM_OOV_BUCKETS = cover_constants.NUM_OOV_BUCKETS
_transformed_name = cover_constants.transformed_name

def preprocessing_fn(inputs):

    features_dict = {}

    ### START CODE HERE ###

    for feature in _NUMERIC_FEATURE_KEYS:
        scaled = tft.scale_to_0_1(inputs[feature])
        features_dict[_transformed_name(feature)] = tf.reshape(scaled, [-1])

    for feature in _SCALE_Z_FEATURE_KEYS:
        data_col = inputs[feature] 
        # Transform using scaling to z score
        # Hint: tft.scale_to_z_score
        features_dict[_transformed_name(feature)] = tf.reshape(tft.scale_to_z_score(data_col), [-1])

    for feature, vocab_size in _VOCAB_FEATURE_DICT.items():
        data_col = tft.compute_and_apply_vocabulary(inputs[feature], num_oov_buckets = _NUM_OOV_BUCKETS) 
        one_hot = tf.one_hot(data_col, vocab_size + _NUM_OOV_BUCKETS)
        features_dict[_transformed_name(feature)] = tf.reshape(one_hot, [-1, vocab_size + _NUM_OOV_BUCKETS])
        # Transform using vocabulary available in column
        # Hint: Use tft.compute_and_apply_vocabulary
        # features_dict[_transformed_name(feature)] = tf.reshape(tft.compute_and_apply_vocabulary(data_col), [-1])

#    for feature in _HASH_STRING_FEATURE_KEYS:
#        data_col = inputs[feature] 
#        # Transform by hashing strings into buckets
#        # Hint: Use tft.hash_strings with the param hash_buckets set to 10
#        features_dict[_transformed_name(feature)] = tft.hash_strings(data_col, hash_buckets=10)
    
    ### END CODE HERE ###  

    # No change in the label
    features_dict[_LABEL_KEY] = inputs[_LABEL_KEY]

    return features_dict
