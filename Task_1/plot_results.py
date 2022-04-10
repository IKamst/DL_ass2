###
# plot_results.py
# Deep Learning Assignment 2
# Group 6
###

import os
import pandas as pd
from matplotlib import pyplot as plt

# The run number which can be set to load the output data from different folders
RUN_NUMBER = 3

# Make a plot of the loss over time
def make_loss_plot():

    number_of_files = 8 * 4
    number_of_steps = 100

    total_content_loss_array = [0] * number_of_steps
    total_style_loss_array = [0] * number_of_steps

    directory = 'loss/' + str(RUN_NUMBER)

    # Go through the csv files in the directory and add all of the style and content losses togheter
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if filename.startswith("content_loss"):
            content_loss_array = pd.read_csv(f).values.flatten().tolist()
            total_content_loss_array = [sum(x) for x in zip(content_loss_array, total_content_loss_array)]
        elif filename.startswith("style_loss"):
            style_loss_array = pd.read_csv(f).values.flatten().tolist()
            total_style_loss_array = [sum(x) for x in zip(style_loss_array, total_style_loss_array)]

    # Calculate the mean style and content loss per step
    total_content_loss_array = [x / number_of_files for x in total_content_loss_array]
    total_style_loss_array = [x / number_of_files for x in total_style_loss_array]

    # Calculate the mean total loss per step
    total_loss_array = [sum(x) for x in zip(total_content_loss_array, total_style_loss_array)]

    k = range(1, len(total_content_loss_array) + 1)

    # Create a plot of the style loss, content loss and total loss per step
    plt.figure(figsize=(13, 8))
    plt.plot(k, total_content_loss_array, label="Content loss")
    plt.plot(k, total_style_loss_array, label='Style loss')
    plt.plot(k, total_loss_array, label='Total loss')
    plt.legend(loc="upper right", prop={'size': 20})
    plt.xlabel('Step', fontsize=24)
    plt.ylabel('Loss', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # Change the tile and name of the file if required
    plt.title("Mean loss per step for pre-trained style transfer", fontsize=30)
    plt.savefig('loss_plot_pretrained.png')
