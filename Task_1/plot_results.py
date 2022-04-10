import os
from operator import add

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def make_loss_plot():
    number_of_files = 8 * 4
    number_of_steps = 100

    total_content_loss_array = [0] * number_of_steps
    total_style_loss_array = [0] * number_of_steps
    std_content_loss_tuple = ()
    std_style_loss_tuple = ()

    directory = 'loss/pretrained'

    # Read from csv
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        print(filename)
        if filename.startswith("content_loss"):
            print("hello")
            content_loss_array = pd.read_csv(f).values.flatten().tolist()
            total_content_loss_array = [sum(x) for x in zip(content_loss_array, total_content_loss_array)]
        elif filename.startswith("style_loss"):
            style_loss_array = pd.read_csv(f).values.flatten().tolist()
            total_style_loss_array = [sum(x) for x in zip(style_loss_array, total_style_loss_array)]

    total_content_loss_array = [x / (number_of_files * 10000000000) for x in total_content_loss_array]
    # std_content_loss_array = np.std((a, b, c), axis=0, ddof=1)
    total_style_loss_array = [x / (number_of_files * 10000000000) for x in total_style_loss_array]

    total_loss_array = [sum(x) / 10000000000 for x in zip(total_content_loss_array, total_style_loss_array)]

    k = range(1, len(total_content_loss_array) + 1)

    # HR@K plot
    plt.figure(figsize=(13, 8))
    plt.plot(k, total_content_loss_array, label="Content loss")
    plt.plot(k, total_style_loss_array, label='Style loss')
    plt.plot(k, total_loss_array, label='Total loss')
    plt.legend(loc="upper right", prop={'size': 20})
    plt.xlabel('Step', fontsize=24)
    plt.ylabel('Loss (x$10^{10}$)', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("Mean loss per step for pre-trained style transfer", fontsize=30)
    plt.savefig('loss_plot_pretrained.png')
