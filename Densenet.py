import os
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D, Dense, Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D


def dense_block(x, blocks, name):
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias=False,
               name=name + '_conv')(x)
    # x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name):
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1, use_bias=False,
                name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3, padding='same', use_bias=False,
                name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def cnn_model(img_input, num_classes=None):
    # eps = 1.1e-5
    skip_connections = []
    # 处理尺寸不同的后端
    global bn_axis ,conv1,conv2,conv3,conv4,conv5

    bn_axis = 3

    conv1 = img_input
    skip_connections.append(conv1)

    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='conv1/bn')(x)
    x = Activation('relu', name='conv1/relu')(x)

    conv2 = x
    skip_connections.append(conv2)

    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)
    # DenseNet201 ： dense_block == [6, 12, 48, 32]
    x = dense_block(x, 6, name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    conv3 = x
    skip_connections.append(conv3)
    x = AveragePooling2D(2, strides=2, name='pool2' + '_pool')(x)

    x = dense_block(x, 12, name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    conv4 = x
    skip_connections.append(conv4)
    x = AveragePooling2D(2, strides=2, name='pool3' + '_pool')(x)

    x = dense_block(x, 24, name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    conv5 = x
    x = AveragePooling2D(2, strides=2, name='pool4' + '_pool')(x)

    # skip_connections.append(conv5)
    # x = dense_block(x, 16, name='conv5')
    #
    # x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
    #                        name='bn')(x)



    return conv5,skip_connections

if __name__ == "__main__":
    # model = cnn_model((256, 256, 3))
    # model.summary()
    # names = ["input_1", "conv1/relu", "pool2_conv", "pool3_conv"]
    print(cnn_model((256, 256, 3)))


