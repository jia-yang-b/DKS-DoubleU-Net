# coding=utf-8
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, \
    Activation, ZeroPadding2D
from tensorflow.keras.layers import add, Flatten
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def identity_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

def bottleneck_Block(inpt,nb_filters,strides=(1,1),with_conv_shortcut=False):
    k1,k2,k3=nb_filters
    x = Conv2d_BN(inpt, nb_filter=k1, kernel_size=1, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=k2, kernel_size=3, padding='same')
    x = Conv2d_BN(x, nb_filter=k3, kernel_size=1, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=k3, strides=strides, kernel_size=1)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

# def resnet_34(width,height,channel,classes):
#     inpt = Input(shape=(width, height, channel))
#     x = ZeroPadding2D((3, 3))(inpt)
#
#     #conv1
#     x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
#     x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
#
#     #conv2_x
#     x = identity_Block(x, nb_filter=64, kernel_size=(3, 3))
#     x = identity_Block(x, nb_filter=64, kernel_size=(3, 3))
#     x = identity_Block(x, nb_filter=64, kernel_size=(3, 3))
#
#     #conv3_x
#     x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
#     x = identity_Block(x, nb_filter=128, kernel_size=(3, 3))
#     x = identity_Block(x, nb_filter=128, kernel_size=(3, 3))
#     x = identity_Block(x, nb_filter=128, kernel_size=(3, 3))
#
#     #conv4_x
#     x = identity_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
#     x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
#     x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
#     x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
#     x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
#     x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
#
#     #conv5_x
#     x = identity_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
#     x = identity_Block(x, nb_filter=512, kernel_size=(3, 3))
#     x = identity_Block(x, nb_filter=512, kernel_size=(3, 3))
#     x = AveragePooling2D(pool_size=(7, 7))(x)
#     x = Flatten()(x)
#     x = Dense(classes, activation='softmax')(x)
#
#     model = Model(inputs=inpt, outputs=x)
#     return model

def resnet_50(x):
    skip_connections = []
    # inpt = Input(inputs)
    skip_connections.append(x)
    x = ZeroPadding2D((3, 3))(x)
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    skip_connections.append(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    #conv2_x
    x = bottleneck_Block(x, nb_filters=[64,64,256],strides=(1,1),with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[64,64,256])
    x = bottleneck_Block(x, nb_filters=[64,64,256])
    skip_connections.append(x)

    #conv3_x
    x = bottleneck_Block(x, nb_filters=[128, 128, 512],strides=(2,2),with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])
    skip_connections.append(x)

    #conv4_x
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024],strides=(2,2),with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])


    #conv5_x
    # x = bottleneck_Block(x, nb_filters=[512, 512, 2048], strides=(2, 2), with_conv_shortcut=True)
    # x = bottleneck_Block(x, nb_filters=[512, 512, 2048])
    # x = bottleneck_Block(x, nb_filters=[512, 512, 2048])


    # x = AveragePooling2D(pool_size=(7, 7))(x)
    # x = Flatten()(x)
    # # x = Dense(classes, activation='softmax')(x)
    #
    # model = Model(inputs=inpt, outputs=x)
    return x,skip_connections

if __name__ == "__main__":
    model = resnet_50((512, 512, 3))
    model.summary()