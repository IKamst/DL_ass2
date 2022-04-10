###
# main1.py
# Deep Learning Assignment 2
# Group 6
###

import data_analysis
import data_handling
import style_transfer
import plot_results

if __name__ == "__main__":
    # Analyse the data
    # images_train, labels_train = data_analysis.perform_data_analysis()
    # # Get the data and create a training set and validation set of the content images
    # train_ds, validation_ds = data_handling.create_train_validation_ds()
    # # Perform style transfer
    # style_transfer.main_style_transfer(train_ds, validation_ds)
    # Make plots of the loss over time
    plot_results.make_loss_plot()
