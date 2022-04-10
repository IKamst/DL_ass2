###
# data_analysis.py
# Deep Learning Assignment 2
# Group 6
###

from pathlib import Path
import cv2
from matplotlib import pyplot as plt

# Plot the images.
def plot_images(image):
    show_images = False
    if show_images:
        cv2.imshow('image', image)
    return

# Keep track of the images and their labels
def process_data(directory, list_cnt_images):
    cnt_images = 0
    names_images = [x for x in directory.iterdir()]
    class_labels = []
    class_images = []
    # Loop over all images
    for name in names_images:
        # Keep track of the amount of images
        cnt_images = cnt_images + 1
        # Read the image.
        img = cv2.imread(str(name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Plot the image
        plot_images(img)
        # Keep track of the class labels and the images
        class_labels.append(directory.name)
        class_images.append(img)
    list_cnt_images.append(cnt_images)
    return list_cnt_images, class_labels, class_images

# Plot the amount of images per class
def plot_images_per_class(subdir_names, list_cnt_images):
    plt.bar(subdir_names, list_cnt_images)
    plt.xlabel('Class')
    plt.ylabel('Number of images')
    plt.title('Number of images per class')
    plt.show()
    return

# Get the images and their labels
def read_images(subdir, labels, images, list_cnt_images):
    # Loop over the subdirectories
    for directory in subdir:
        list_cnt_images, class_labels, class_images = \
            process_data(directory, list_cnt_images)
        labels += class_labels
        images += class_images
    return images, labels, list_cnt_images

# Perform data analysis on the content images
def perform_data_analysis():
    # Select the path of the image folder and its subdirectories
    image_path_train = Path("Data/content_images/train")
    # Determine the subdirectories and their names
    subdir_train = [x for x in image_path_train.iterdir() if x.is_dir()]
    subdir_names = [x.name for x in image_path_train.iterdir() if x.is_dir()]
    images_train, labels_train, list_cnt_images = read_images(subdir_train, [], [], [])
    plot_images_per_class(subdir_names, list_cnt_images)
    return images_train, labels_train