# Data analysis for the images.
# TODO update more for Deep Learning
# TODO doe wee need min_x min_y

# importing modules
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import tensorflow as tf

# Load the image in the correct way, so it can be used as input for the model.
def load_image(content_path):
    # content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg',
    #                                       'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
    img = tf.io.read_file(content_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [224, 224])
    img = img[tf.newaxis, :]
    return img


# Keep track of the images and their labels.
def process_data(directory, list_cnt_images):
    cnt_images = 0
    # names_images = [x for x in directory.iterdir()]
    class_labels = []
    class_images = []
    # Loop over all images.
    for file_name in os.listdir(directory):
        name = os.path.join(directory, file_name)
        # Keep track of the amount of images.
        cnt_images = cnt_images + 1
        img = load_image(name)
        print(img)
        tensor_image = tf.Variable(img)
        print(tensor_image)
        # Keep track of the class labels and the images.
        class_labels.append(directory.name)
        class_images.append(tensor_image)
    list_cnt_images.append(cnt_images)
    return list_cnt_images, class_labels, class_images


# Plot the amount of images per class.
def plot_images_per_class(subdir_names, list_cnt_images):
    plt.bar(subdir_names, list_cnt_images)
    plt.xlabel('Class')
    plt.ylabel('Number of images')
    plt.title('Number of images per class')
    plt.show()
    return


# Scatter-plot of the dimensions of the images.
def plot_dimension_of_images(dimension_array, subdir_names):
    for dimension_class in dimension_array:
        plt.scatter(dimension_class[0], dimension_class[1])
    plt.xlabel('Width of image')
    plt.ylabel('Height of image')
    plt.title('Dimension of images')
    plt.legend(subdir_names)
    plt.show()
    return


def read_images(subdir, labels, images, list_cnt_images):
    # Loop over the subdirectories.
    for directory in subdir:
        list_cnt_images, class_labels, class_images = \
            process_data(directory, list_cnt_images)
        labels += class_labels
        images += class_images
    return images, labels, list_cnt_images


# Data analysis.
def perform_data_analysis():
    # Select the path of the image folder and its subdirectories.
    image_path_train = Path("Data/content_images/train")
    # Determine the subdirectories and their names.
    subdir_train = [x for x in image_path_train.iterdir() if x.is_dir()]
    print(subdir_train)
    subdir_names = [x.name for x in image_path_train.iterdir() if x.is_dir()]
    images_train, labels_train, list_cnt_images = read_images(subdir_train, [], [], [])
    print(list_cnt_images)
    plot_images_per_class(subdir_names, list_cnt_images)
    return images_train, labels_train
