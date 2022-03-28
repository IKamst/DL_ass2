import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

AUTOTUNE = tf.data.AUTOTUNE


def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def load_data():
    # content_images = tfds.load("clic")
    content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg',
                                           'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
    style_path = tf.keras.utils.get_file('starry.jpg',
                                         'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1200px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg')
    content_image = load_img(content_path)
    style_image = load_img(style_path)
    return content_image, style_image


def process_data(images, labels):
    if len(images) != len(labels):
        print("data_x not of same size as data_y")
        sys.exit()

    return train_test_split(images, labels, test_size=0.2, stratify=labels)


def create_train_test_ds():
    # TODO change seed
    data_dir = Path("data")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        batch_size=32,
        image_size=(224, 224))

    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        batch_size=32,
        image_size=(224, 224))

    print(train_ds)
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    normalised_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalised_train_ds))
    print(labels_batch)
    normalised_test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalised_test_ds))
    # new_train_ds = normalised_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    # new_test_ds = normalised_test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return normalised_train_ds, normalised_test_ds
