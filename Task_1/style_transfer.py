# Original style transfer approach based on Gatys et al. 'A Neural Algorithm of Artistic Style'
# Based on the explanation as given in https://www.tensorflow.org/tutorials/generative/style_transfer

# Import and configure modules
import PIL
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import IPython.display as display


# Make a model that returns the style and content tensors.
class StyleContentModel(tf.keras.models.Model):
    # Initialise the model.
    def __init__(self, style_layers, content_layers, train_ds, test_ds):
        super(StyleContentModel, self).__init__()
        self.vgg = make_vgg_layers(style_layers + content_layers, train_ds, test_ds)
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
        # TODO determine what exactly happens here.
        content_dict = {content_name: value for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


# Create the content and style layers using the approach defined in the paper by Gatys et al.
# The intermediate layers of the VGG19 network represent the style and content.
def create_content_style_layers():
    content_layers = ['block5_conv2']

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)
    return content_layers, style_layers, num_content_layers, num_style_layers


# Build a VGG19 model and return a list of intermediate layer outputs.
def make_vgg_layers(layer_names, train_ds, test_ds):
    # If weights is set to 'imagenet', then a pretrained VGG is loaded.
    # If weights is set to NONE, then random weights are used.
    # TODO do we include_top?
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    # Setting trainable to True allows the model to learn and change weights.
    vgg.trainable = True

    vgg.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # tf.Tensor(train_x)
    # print(train_x)
    # print(train_y)
    # train_x = np.asarray(train_x).astype(np.float)
    # train_y = np.asarray(train_y).astype(np.float)
    # vgg.fit(train_x, train_y)
    vgg.fit(
        train_ds,
        validation_data=test_ds,
        epochs=3
    )
    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


# Calculate style
# TODO understand this better.
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def style_content_loss(outputs, num_style_layers, num_content_layers, style_targets, content_targets):
    style_weight = 1e4
    content_weight = 1e-2
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss


# Update the image
# TODO check if tf.function() is necessary in any way?
# @tf.function()
def train_step(image, extractor, opt, num_style_layers, num_content_layers, style_targets, content_targets):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, num_style_layers, num_content_layers, style_targets, content_targets)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
    return


# Create an image from the tensor.
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)
    return


def train_style_transfer(image, extractor, opt, num_style_layers, num_content_layers, style_targets, content_targets, epochs, steps_per_epoch):
    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(image, extractor, opt, num_style_layers, num_content_layers, style_targets, content_targets)
            print(".", end='', flush=True)
        tensor_to_image(image).show()
        print("Train step: {}".format(step))


# Main function. This calls all other functions that are used during the style transfer process.
def main_style_transfer(train_ds, test_ds):
    content_layers, style_layers, num_content_layers, num_style_layers = create_content_style_layers()
    # Create the model.
    extractor = StyleContentModel(style_layers, content_layers, train_ds, test_ds)
    # Set style and content targets.
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']
    image = tf.Variable(content_image)

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    epochs = 10
    steps_per_epoch = 10

    train_style_transfer(image, extractor, opt, num_style_layers, num_content_layers, style_targets, content_targets, epochs, steps_per_epoch)
    return
