from pathlib import Path
import tensorflow as tf
from matplotlib import pyplot as plt

from style_transfer import tensor_to_image

IMAGE_SIZE = [224, 224] #Default image size for VGG19


def load_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    new_shape = tf.cast([224, 224], tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def get_style_image():
    style_path = tf.keras.utils.get_file('starry.jpg',
                                         'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1200px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg')

    style_path = tf.keras.utils.get_file('kandinsky5.jpg',
                                         'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

    style_image = load_img(style_path)
    # print("Style Images")
    # print(style_image)
    # style_image = style_image[None, :]
    # print(style_image)
    return style_image


def create_train_test_ds():
    # TODO change seed
    data_dir = Path("Data/content_images/train")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        batch_size=32,
        image_size=(224, 224))

    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        batch_size=32,
        image_size=(224, 224))

    # for images, labels in test_ds.take(1):
    #     content_image = images[0]
    #     plt.imshow(content_image.numpy().astype("uint8"))
    #     plt.show()

    return train_ds, test_ds
