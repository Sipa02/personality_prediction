"""
Author: Syifa Azzahro
Date: 18/8/2025
This is transform.py module.
Usage:
- This module is used to define the transformation logic for the TFX pipeline.
"""

import tensorflow as tf
import tensorflow_transform as tft

CATEGORICAL_FEATURES = {
    "Stage_fear": 2,
    "Drained_after_socializing": 2,

}

NUMERICAL_FEATURES = [
    "Time_spent_Alone",
    "Social_event_attendance",
    "Going_outside",
    "Friends_circle_size",
    "Post_frequency",
]

LABEL_KEY = "Personality"


def transformed_name(key):
    """Transform feature key

    Args:
        key (str): the key to be transformed

    Returns:
        str: transformed key
    """

    return f"{key}_xf"


def convert_num_to_one_hot(label_tensor, num_labels=2):
    """Convert a label (0 or 1) into a one-hot vector

    Args:
        label_tensor (int): label tensor (0 or 1)
        num_labels (int, optional): num of label. Defaults to 2.

    Returns:
        tf.Tensor: label tensor
    """

    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])



def preprocessing_fn(inputs):
    """Preprocess input features into transformed features

    Args:
        inputs (dict): map from feature keys to raw features

    Returns:
        dict: map from features keys to transformed features
    """

    outputs = {}

    for key, dim in CATEGORICAL_FEATURES.items():
        int_value = tft.compute_and_apply_vocabulary(
            inputs[key], top_k=dim + 1
        )
        outputs[transformed_name(key)] = convert_num_to_one_hot(
            int_value, num_labels=dim + 1
        )
    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_0_1(inputs[feature])
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    return outputs
