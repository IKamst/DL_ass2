###
# style_transfer.py
# Deep Learning Assignment 2
# Group 6
###

# Original style transfer approach based on Gatys et al. 'A Neural Algorithm of Artistic Style'
# Based on the explanation as given in https://www.tensorflow.org/tutorials/generative/style_transfer

import PIL
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import re
import model_style_transfer

# The number of classes in our content image data
N_CLASSES = 8
# The run number which can be set to save the output data in different folders
RUN_NUMBER = 4

# Build a VGG19 model and return a list of intermediate layer outputs
def make_vgg_layers(layer_names, train_ds, validation_ds):

    # If weights is set to 'imagenet', then a pretrained VGG is loaded
    # If weights is set to None, then random weights are used (non pre-trained)
    image_input = tf.keras.layers.Input(shape=(224, 224, 3))
    vgg_base_model = tf.keras.applications.VGG19(include_top=False, weights=None, input_tensor=image_input)
    vgg_base_model.summary()

    # Create new layers, so the model can train and classify images from our content image dataset
    FC_layer_Flatten = tf.keras.layers.Flatten()(vgg_base_model.output)
    Classification = tf.keras.layers.Dense(units=N_CLASSES, activation='softmax')(FC_layer_Flatten)
    model = tf.keras.Model(inputs=image_input, outputs=Classification)
    model.summary()

    # Train the model
    base_learning_rate = 0.00001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    # Show the performance of the model
    history = model.fit(train_ds, epochs=10, batch_size=32, validation_data=validation_ds)
    print(history)

    # Get the outputs of the layers
    outputs = [model.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([model.input], outputs)
    return model

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

# Calculate style, which is described by the means and correlations across the different feature maps.
# This can be done using the Gram matrix.
def gram_matrix(feature_map):
    result = tf.linalg.einsum('bijc,bijd->bcd', feature_map, feature_map)
    input_shape = tf.shape(feature_map)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

# Clip the values of the image, so they are between 0 and 1
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

# Determine the loss, which is a linear combination of the style and content loss
def style_content_loss(outputs, num_style_layers, num_content_layers, style_targets, content_targets,
                       content_loss_array, style_loss_array):

    # Set weights that change the influence of the style and content in the updated image
    style_weight = 0.5
    content_weight = 1e4

    # Get the current outputs for style and content.
    style_outputs = outputs['style']
    content_outputs = outputs['content']

    # Determine the style loss
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers
    style_loss_array.append(style_loss.numpy())

    # Determine the content loss
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    content_loss_array.append(content_loss.numpy())
    loss = style_loss + content_loss

    return loss, content_loss_array, style_loss_array

# Update the image using gradient descent, the content loss and the style loss
def train_step(image, style_transfer_model, opt, num_style_layers, num_content_layers, style_targets, content_targets,
               content_name, style_name, content_loss_array, style_loss_array):
    with tf.GradientTape() as tape:
        outputs = style_transfer_model(image)
        loss, content_loss_array, style_loss_array = style_content_loss(outputs, num_style_layers, num_content_layers,
                                                                        style_targets, content_targets, content_name,
                                                                        style_name, content_loss_array, style_loss_array)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

    return image, content_loss_array, style_loss_array

# Create an image from a tensor
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# Plot and save the image
def imshow(image, step):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    plt.show()

    path = 'saved_images/' + str(RUN_NUMBER)
    plt.savefig(path + '/' + str(step))

    return

# Train for the style transfer to generate an image
def train_style_transfer(image, style_transfer_model, opt, num_style_layers, num_content_layers, style_targets,
                         content_targets, epochs, steps_per_epoch, content_name, style_name):
    step = 0

    content_loss_array = []
    style_loss_array = []

    # Train the model for a certain amount of epochs and report the outputs
    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            step += 1
            image, content_loss_array, style_loss_array = \
                train_step(image, style_transfer_model, opt, num_style_layers, num_content_layers, style_targets,
                           content_targets, content_name, style_name, content_loss_array, style_loss_array)
            print(".", end='', flush=True)
        # Reformat the image, so it can be shown
        image_show = image[0, :, :, :]
        plt.imshow(tensor_to_image(image_show))
        # Save the images
        path = 'saved_images/' + str(RUN_NUMBER)
        plt.savefig(path + '/' + content_name + style_name + str(step))
        print("Train step: {}".format(step))

    # Save the loss
    path = 'loss/' + str(RUN_NUMBER) + "/"
    try:
        os.makedirs(path)
    except OSError:
        print("Directory already exists")
    np.savetxt(path + "content_loss_" + content_name + "_" + style_name + ".csv", content_loss_array, delimiter=",")
    np.savetxt(path + "style_loss_" + content_name + "_" + style_name + ".csv", style_loss_array, delimiter=",")
    return

# Load the image in the correct way, so it can be used as input for the model
def load_image(content_path):
    img = tf.io.read_file(content_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [224, 224])
    img = img[tf.newaxis, :]
    return img

# Initialise some variables for style transfer and then train the style transfer model.
def transfer_style(style_transfer_model, style_image, content_image, num_content_layers, num_style_layers, content_name,
                   style_name):

    # Set style and content targets.
    style_targets = style_transfer_model(style_image)['style']
    content_targets = style_transfer_model(content_image)['content']

    # Train and perform style transfer
    image = tf.Variable(content_image)
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    epochs = 10
    steps_per_epoch = 10
    train_style_transfer(image, style_transfer_model, opt, num_style_layers, num_content_layers, style_targets, content_targets,
                         epochs, steps_per_epoch, content_name, style_name)
    return

# Loop over the content and style images, so we run style transfer using the same trained CNN
def test_style_transfer(style_transfer_model, num_content_layers, num_style_layers):

    # Loop over the content test images
    directory_test = "Data/content_images/test"
    for filename in os.listdir(directory_test):
        content_name = os.path.join(directory_test, filename)
        content_image = load_image(content_name)
        imshow(content_image, 0)
        content_image = tf.Variable(content_image)

        # For that test image, run style transfer with all style images
        directory_style = "Data/style_images"
        for name in os.listdir(directory_style):
            style_name = os.path.join(directory_style, name)
            style_image = load_image(style_name)
            imshow(style_image, 0)
            filename = re.split('.jpg', filename)[0]
            name = re.split('.jpg', name)[0]
            transfer_style(style_transfer_model, style_image, content_image, num_content_layers, num_style_layers,
                           filename, name)
    return

# Perform style transfer by first training a CNN model and then using the output of its layers to extract style and
# content. Then, mix the style and content to generate a new image.
def main_style_transfer(train_ds, validation_ds):

    # Make a directory to save the generated images
    path = 'saved_images/' + str(RUN_NUMBER)
    try:
        os.makedirs(path)
    except OSError:
        print("Directory already exists")

    # Set the layers that are used to extract the content and style from
    content_layers, style_layers, num_content_layers, num_style_layers = create_content_style_layers()

    # Create the style transfer model
    style_transfer_model = model_style_transfer.StyleTransferModel(style_layers, content_layers, train_ds, validation_ds)

    # Use the style transfer model to generate images
    test_style_transfer(style_transfer_model, num_content_layers, num_style_layers)

    return
