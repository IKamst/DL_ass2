import tensorflow as tf
import tensorflow_datasets as tfds

def load_data():
    content_images = tfds.load("clic")