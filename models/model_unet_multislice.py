import sys

import os
from keras import backend as K
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Dropout, Activation, Permute
from keras.layers import Input, Conv2D, UpSampling2D
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model
from tensorflow.python.client import device_lib

from helpers.metrics import multislice_dice_loss, multislice_dice_coef, sensitivity_specificity_loss, \
    multislice_categorical_focal_loss
from helpers.shared_helpers import SharedHelpers

sh = SharedHelpers()

sys.setrecursionlimit(10000)
K.set_image_data_format("channels_last")

"""Building a U-Net. that can take multiple slices as channles"""


def get_model(slab_size, base_dir="", save_model=True):
    """
    Get the UNET model
    """
    sh.print("UNET-MULTISLICE")
    local_device_protos = device_lib.list_local_devices()
    list_of_gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    NUM_GPUS = len(list_of_gpus)

    model = build_unet(image_width=slab_size[0], image_height=slab_size[1], num_slices=slab_size[2])

    if save_model:
        save_template_model(model, base_dir)

    gpu_model = multi_gpu_model(model, gpus=NUM_GPUS)
    sh.print('LR', 0.01)
    gpu_model.compile(optimizer=Adam(lr=0.001), loss=multislice_dice_loss, metrics=[multislice_dice_coef])
    # gpu_model.compile(optimizer=Adam(lr=0.001), loss=sensitivity_specificity_loss)
    sh.print(model.summary(), gpu_model.summary())

    return model, gpu_model


def save_template_model(model, base_dir):
    """
    Save model to the experiment directory
    :param model: Training model
    :return: None
    """
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(base_dir, "model.json"), "w") as json_file:
        json_file.write(model_json)
    sh.print("Model Saved")


def build_unet(base_num_filters=16, convs_per_depth=2, kernel_size=3, num_classes=2,
               image_height=512, image_width=512, activation='relu', dropout_rate=0.2,
               net_depth=4, conv_order='conv_first', use_batch_normalization=True, num_slices=9):
    """Builds a U-Net.
    """

    if image_height / 2 ** net_depth == 0 or image_width / 2 ** net_depth == 0:
        raise RuntimeError('Mismatch between image size and network depth.')

    # Get channel axis
    axis = get_channel_axis()

    # conv with some arguments assigned
    def local_conv(num_filt):
        return conv(num_filt, kernel_size=kernel_size, activation=activation, conv_order=conv_order,
                    use_batch_normalization=use_batch_normalization)

    contr_tensors = dict()  # For the concatenation during expansion. key: depth; val: tensor

    def contract(depth):
        """Returns a contraction layer."""

        def composite_layer(x):
            num_filt = base_num_filters * 2 ** depth
            for _ in range(convs_per_depth):
                x = local_conv(num_filt)(x)
            contr_tensors[depth] = x
            x._keras_history[0].name = 'contr_%d' % depth
            x = MaxPooling2D()(x)
            return x

        return composite_layer

    def expand(depth):
        """Returns an expansion layer."""

        def composite_layer(x, tensor_contr):
            """
            Arguments:
            x: tensor from the previous layer.
            tensor_contr: tensor from a contraction layer.
            """
            num_filt = base_num_filters * 2 ** depth
            x = Concatenate(axis=axis, name='concat_%d' % depth)([UpSampling2D()(x), tensor_contr])
            for _ in range(convs_per_depth):
                x = local_conv(num_filt)(x)
            x._keras_history[0].name = 'expand_%d' % depth
            return x

        return composite_layer

    if K.image_data_format() == 'channels_first':
        inputs = Input((num_slices, image_height, image_width))
    else:
        inputs = Input((image_height, image_width, num_slices))

    tensor = inputs

    # Contracting path
    for i in range(net_depth):
        tensor = contract(i)(tensor)

    # Deepest layer
    num_filters = base_num_filters * 2 ** net_depth
    if dropout_rate:
        tensor = Dropout(dropout_rate)(tensor)
    for i in range(convs_per_depth):
        tensor = local_conv(num_filters)(tensor)
    tensor._keras_history[0].name = 'encode'

    # Expanding path
    for i in reversed(range(net_depth)):
        tensor = expand(i)(tensor, contr_tensors[i])

    # Segmentation layer and loss function related
    tensor = Conv2D(num_classes, 1, activation=None)(tensor)
    if K.image_data_format() == 'channels_first':
        tensor = Permute((2, 3, 1))(tensor)
        tensor = Activation('softmax')(tensor)
        tensor = Permute((3, 1, 2), name='segmentation')(tensor)
    else:
        tensor = Activation('softmax', name='segmentation')(tensor)

    # tensor = Reshape((image_height * image_width, num_classes))(tensor)

    model = Model(inputs=inputs, outputs=tensor)

    return model


def get_channel_axis():
    """Gets the channel axis."""
    if K.image_data_format() == 'channels_first':
        return 1
    else:
        return 3


def conv(num_filters, kernel_size=3, activation='relu', conv_order='conv_first', use_batch_normalization=True):
    """Returns a composite layer for Conv2D with BatchNormalization.
    """

    def composite_layer(tensor):
        if conv_order == 'conv_first':
            tensor = Conv2D(num_filters, kernel_size, padding='same')(tensor)
        if use_batch_normalization == True:
            tensor = BatchNormalization(axis=get_channel_axis())(tensor)
        tensor = Activation(activation)(tensor)
        if conv_order == 'conv_last':
            tensor = Conv2D(num_filters, kernel_size, padding='same')(tensor)
        return tensor

    return composite_layer
