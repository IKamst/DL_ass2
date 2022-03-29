from pathlib import Path
import tensorflow as tf

IMAGE_SIZE = [224, 224] #Default image size for VGG19


def load_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    new_shape = tf.cast([224, 224], tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def load_data():
    # content_images = tfds.load("clic")
    content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg',
                                           'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
    style_path = tf.keras.utils.get_file('starry.jpg',
                                         'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1200px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg')
    content_image = load_img(content_path)
    style_image = load_img(style_path)
    return content_image, style_image


def get_style_image():
    style_path = tf.keras.utils.get_file('starry.jpg',
                                         'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1200px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg')
    style_image = load_img(style_path)
    # print("Style Images")
    # print(style_image)
    # style_image = style_image[None, :]
    # print(style_image)
    return style_image

def create_train_test_ds():
    # TODO change seed
    data_dir = Path("data")
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

    print(train_ds)
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break
    return train_ds, test_ds

