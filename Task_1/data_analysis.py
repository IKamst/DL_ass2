# Data analysis for the images.
# TODO update more for Deep Learning
# TODO doe wee need min_x min_y

# importing modules
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt


# Plot the images.
def plot_images(image):
    show_images = False
    if show_images:
        cv2.imshow('image', image)
    return


# Determine the dimensions, while keeping track of the images and their labels.
def determine_dimensions(directory, dimension_array, list_cnt_images):
    array_x_dim = []
    array_y_dim = []
    cnt_images = 0
    names_images = [x for x in directory.iterdir()]
    class_labels = []
    class_images = []
    min_x_dim = np.inf
    min_y_dim = np.inf
    # Loop over all images.
    for name in names_images:
        # Keep track of the amount of images.
        cnt_images = cnt_images + 1
        # Read the image.
        img = cv2.imread(str(name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Keep track of the dimensions of the image.
        array_x_dim.append(img.shape[1])
        array_y_dim.append(img.shape[0])
        if img.shape[1] < min_x_dim:
            min_x_dim = img.shape[1]
        if img.shape[0] < min_y_dim:
            min_y_dim = img.shape[0]
        # Plot the image.
        plot_images(img)
        # Keep track of the class labels and the images.
        class_labels.append(directory.name)
        class_images.append(img)
    list_cnt_images.append(cnt_images)
    dimension_array.append([array_x_dim, array_y_dim])
    return dimension_array, list_cnt_images, class_labels, class_images, min_x_dim, min_y_dim


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


# Data analysis.
def perform_data_analysis():
    # Select the path of the image folder and its subdirectories.
    image_folder = Path("data")
    # Determine the subdirectories and their names.
    subdir = [x for x in image_folder.iterdir() if x.is_dir()]
    subdir_names = [x.name for x in image_folder.iterdir() if x.is_dir()]
    # Initialise some variables.
    list_cnt_images = []
    dimension_array = []
    labels = []
    images = []
    min_x = np.inf
    min_y = np.inf
    # Loop over the subdirectories.
    for directory in subdir:
        dimension_array, list_cnt_images, class_labels, class_images, min_x_dim, min_y_dim = \
            determine_dimensions(directory, dimension_array, list_cnt_images)
        if min_x_dim < min_x:
            min_x = min_x_dim
        if min_y_dim < min_y:
            min_y = min_y_dim
        labels += class_labels
        images += class_images
    plot_images_per_class(subdir_names, list_cnt_images)
    plot_dimension_of_images(dimension_array, subdir_names)
    return labels, images, min_x, min_y
