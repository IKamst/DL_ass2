###
# data_handling.py
# Deep Learning Assignment 2
# Group 6
###

from pathlib import Path
import tensorflow as tf

# Get the content image data and split it into a training and validation set
def create_train_validation_ds():
    data_dir = Path("Data/content_images/train")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        batch_size=32,
        image_size=(224, 224))

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        batch_size=32,
        image_size=(224, 224))

    return train_ds, validation_ds
