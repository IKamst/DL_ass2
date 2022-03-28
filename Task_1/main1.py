###
# main1.py
# Deep Learning Assignment 2
# ................
###
from pathlib import Path

import data_analysis
import data_handling
import style_transfer_fast, style_transfer
import tensorflow as tf

if __name__ == "__main__":
    labels, images, min_x, min_y = data_analysis.perform_data_analysis()
    train_ds, test_ds = data_handling.create_train_test_ds()
    # train_x, test_x, train_y, test_y = data_handling.process_data(images, labels)
    # data_handling.load_data()
    # content_image, style_image = data_handling.load_data()
    # style_transfer_fast.fast_transfer_style()
    style_transfer.main_style_transfer(train_ds, test_ds)
