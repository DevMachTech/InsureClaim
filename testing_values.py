import tensorflow as tf

raw_data = [
    {
        "Customer Id": "H5053",
        "Building Dimension": 680,
        "Insured_Period": 1.0,
        "Building_Type": 1,
        "Building_Painted": "V",
        "Building_Fenced": "N",
        "Garden": "O",
        "NumberOfWindows": 3,
        "Geo_Code": 1053,
        "Settlement": "R",
        "Claim": 0,
    }
]

feature_description = {
    "Customer Id": tf.io.FixedLenFeature([], tf.string),
    "Building Dimension": tf.io.FixedLenFeature([], tf.int64),
    "Insured_Period": tf.io.FixedLenFeature([], tf.float32),
    "Building_Type": tf.io.FixedLenFeature([], tf.int64),
    "Building_Painted": tf.io.FixedLenFeature([], tf.string),
    "Building_Fenced": tf.io.FixedLenFeature([], tf.string),
    "Garden": tf.io.FixedLenFeature([], tf.string),
    "NumberOfWindows": tf.io.FixedLenFeature([], tf.int64),
    "Geo_Code": tf.io.FixedLenFeature([], tf.int64),
    "Settlement": tf.io.FixedLenFeature([], tf.string),
    "Claim": tf.io.FixedLenFeature([], tf.int64),
}
