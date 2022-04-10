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
    plot_results.make_loss_plot()
    images_train, labels_train = data_analysis.perform_data_analysis()
    train_ds, test_ds = data_handling.create_train_test_ds()
    style_image = data_handling.get_style_image()
    style_transfer.main_style_transfer(train_ds, test_ds)