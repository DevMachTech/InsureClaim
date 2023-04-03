
SCALE_Z_FEATURE_KEYS = [
        "Building Dimension"
    ]

NUMERIC_FEATURE_KEYS = [
    'Building_Type', 'Insured_Period'
]

# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
VOCAB_SIZE = 10

# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
OOV_SIZE = 10

VOCAB_FEATURE_DICT ={
    'Building_Fenced':2, 'Building_Painted':2, 'Settlement':2
}
# VOCAB_FEATURE_KEYS = ["Building_Fenced","Building_Painted", "Settlement"]

NUM_OOV_BUCKETS = 2

HASH_STRING_FEATURE_KEYS = ["Customer Id"]

LABEL_KEY = "Claim"

# Utility function for renaming the feature
def transformed_name(key):
    return key + '_xf'
