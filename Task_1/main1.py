###
# main1.py
# Deep Learning Assignment 2
# ................
###
import data_handling
import style_transfer_fast, style_transfer

if __name__ == "__main__":
    # data_handling.load_data()
    content_image, style_image = data_handling.load_data()
    # style_transfer_fast.fast_transfer_style()
    style_transfer.main_style_transfer(style_image, content_image)
