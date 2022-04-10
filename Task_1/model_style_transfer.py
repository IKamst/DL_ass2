###
# model_style_transfer.py
# Deep Learning Assignment 2
# Group 6
###

# Based on the explanation as given in https://www.tensorflow.org/tutorials/generative/style_transfer

import tensorflow as tf
import style_transfer

# Make a style transfer model that is able to extract the style and content from an image
class StyleTransferModel(tf.keras.models.Model):

    # Initialise the model
    def __init__(self, style_layers, content_layers, train_ds, validation_ds):
        super(StyleTransferModel, self).__init__()
        self.vgg = style_transfer.make_vgg_layers(style_layers + content_layers, train_ds, validation_ds)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = True

    # The model returns the Gram matrix of the style_layers and content of the content_layers
    def call(self, inputs):
        # Get correctly formatted and preprocessed inputs
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        # Get the output by applying VGG-19 to the input
        outputs = self.vgg(preprocessed_input)
        # Get the weights of the style and content layers
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        # The style output is determined by applying the Gram matrix to the weights of the style layers
        style_outputs = [style_transfer.gram_matrix(style_output) for style_output in style_outputs]

        # Make content and style dictionaries
        content_dict = {content_name: value for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}
