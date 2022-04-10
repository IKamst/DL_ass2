# Original style transfer approach based on Gatys et al. 'A Neural Algorithm of Artistic Style'
# Based on the explanation as given in https://www.tensorflow.org/tutorials/generative/style_transfer

# Import and configure modules
import PIL
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import IPython.display as display
import os
from pathlib import Path
import re

N_CLASSES = 8
RUN_NUMBER = 1


# Make a model that returns the style and content tensors.
class StyleContentModel(tf.keras.models.Model):
    # Initialise the model.
    def __init__(self, style_layers, content_layers, train_ds, validation_ds):
        super(StyleContentModel, self).__init__()
        self.vgg = make_vgg_layers(style_layers + content_layers, train_ds, validation_ds)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = True

    # This model returns the Gram matrix of the style_layers and content of the content_layers.
    def call(self, inputs):
        # Get correctly formatted and preprocessed inputs.
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        # Get the output by applying vgg19 to the input.
        outputs = self.vgg(preprocessed_input)
        # Get the weights of the style and content layers.
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        # The style output is determined by applying the Gram matrix to the weights of the style layers.
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        # Make content and style dictionaries.
        content_dict = {content_name: value for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


# Create the content and style layers using the approach defined in the paper by Gatys et al.
# The intermediate layers of the VGG19 network represent the style and content.
def create_content_style_layers():
    content_layers = ['block4_conv2']

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)
    return content_layers, style_layers, num_content_layers, num_style_layers


# Build a VGG19 model and return a list of intermediate layer outputs.
def make_vgg_layers(layer_names, train_ds, validation_ds):
    # If weights is set to 'imagenet', then a pretrained VGG is loaded.
    # If weights is set to NONE, then random weights are used.
    image_input = tf.keras.layers.Input(shape=(224, 224, 3))
    vgg_base_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_tensor=image_input)
    vgg_base_model.summary()

    # Create new layers, so the model can train and classify images from our dataset
    FC_layer_Flatten = tf.keras.layers.Flatten()(vgg_base_model.output)
    Classification = tf.keras.layers.Dense(units=N_CLASSES, activation='softmax')(FC_layer_Flatten)

    model = tf.keras.Model(inputs=image_input, outputs=Classification)  # Line 12
    model.summary()

    # Train the model
    base_learning_rate = 0.00001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    history = model.fit(train_ds, epochs=5, batch_size=32, validation_data=validation_ds)
    print(history)

    outputs = [model.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([model.input], outputs)
    return model


# Calculate style, which is described by the means and correlations across the different feature maps.
# This can be done using the Gram matrix.
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


# Clip the values of the image, so they are between 0 and 1.
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


# Determine the loss, which is a linear combination of the style and content loss.
def style_content_loss(outputs, num_style_layers, num_content_layers, style_targets, content_targets):
    # Set weights that change the influence of the style and content in the updated image.
    style_weight = 1e-2
    content_weight = 1e4
    # Get the current outputs for style and content.
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    # Determine the style loss.
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    print("STYLE LOSS")
    print(style_loss)
    print("CONTENT LOSS")
    print(content_loss)
    loss = style_loss + content_loss
    print("TOTAL LOSS")
    print(loss)
    return loss


# Update the image using gradient descent with loss.
# @tf.function()
def train_step(image, extractor, opt, num_style_layers, num_content_layers, style_targets, content_targets):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, num_style_layers, num_content_layers, style_targets, content_targets)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
    return image


# Create an image from the tensor.
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


# Plot the image.
def imshow(image, step, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    plt.show()
    path = 'saved_images/' + str(RUN_NUMBER)
    plt.savefig(path + '/' + str(step))
    if title:
        plt.title(title)
    return


# Train the style transfer part.
def train_style_transfer(image, extractor, opt, num_style_layers, num_content_layers, style_targets, content_targets,
                         epochs, steps_per_epoch, content_name, style_name):
    step = 0
    # Train the model using gradient descent for a certain amount of epochs.
    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            step += 1
            image = train_step(image, extractor, opt, num_style_layers, num_content_layers, style_targets,
                               content_targets)
            print(".", end='', flush=True)
        # Reformat the image, so it can be shown.
        image_show = image[0, :, :, :]
        plt.imshow(tensor_to_image(image_show))
        path = 'saved_images/' + str(RUN_NUMBER)
        content_name = re.split('.jpg', content_name)[0]
        style_name = re.split('.jpg', style_name)[0]
        plt.savefig(path + '/' + content_name + style_name + str(step))
        plt.show()
        print("Train step: {}".format(step))
    return


# Load the image in the correct way, so it can be used as input for the model.
def load_image(content_path):
    # content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg',
    #                                       'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
    img = tf.io.read_file(content_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [224, 224])
    img = img[tf.newaxis, :]
    return img


# Initialise some variables for style transfer and then train the style transfer model.
def transfer_style(extractor, style_image, content_image, num_content_layers, num_style_layers, content_name, style_name):
    # Set style and content targets.
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']
    image = tf.Variable(content_image)
    # tensor_to_image(image).show()

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    epochs = 10
    steps_per_epoch = 10
    train_style_transfer(image, extractor, opt, num_style_layers, num_content_layers, style_targets, content_targets,
                         epochs, steps_per_epoch, content_name, style_name)
    return


# Main function. This calls all other functions that are used during the style transfer process.
def main_style_transfer(train_ds, validation_ds):
    # Make a directory to save the images.
    path = 'saved_images/' + str(RUN_NUMBER)
    try:
        os.makedirs(path)
    except OSError:
        print("Directory already exists")

    # Create content and style layers.
    content_layers, style_layers, num_content_layers, num_style_layers = create_content_style_layers()
    # Create the model.
    extractor = StyleContentModel(style_layers, content_layers, train_ds, validation_ds)

    # Loop over images of the test set, so we run style transfer using the same trained CNN for all test data.
    directory_test = "Data/content_images/test"
    for filename in os.listdir(directory_test):
        content_name = os.path.join(directory_test, filename)
        print(content_name)
        content_image = load_image(content_name)
        imshow(content_image, 0)

        # For that test image, run style transfer with all style images.
        directory_style = "Data/style_images"
        for name in os.listdir(directory_style):
            style_name = os.path.join(directory_style, name)
            print(style_name)
            style_image = load_image(style_name)
            imshow(style_image, 0)
            transfer_style(extractor, style_image, content_image, num_content_layers, num_style_layers, filename, name)
    return
