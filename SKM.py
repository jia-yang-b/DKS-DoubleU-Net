import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, \
    Conv2D, Add, Activation, Lambda,Conv1D
# from tflearn.layers.conv import global_avg_pool
from tensorflow.keras import layers


def selective_kernel_layer(concat, middle=4, out_dim=7):
    # sk_conv1 = slim.conv2d(concat, 7, [3, 3], rate=1, activation_fn=lrelu)
    sk_conv1 = Conv2D(7, (3, 3), padding="same")(concat)
    sk_conv2 = Conv2D(7, (5, 5), padding="same")(concat)
    sk_conv3 = Conv2D(7, (7, 7), padding="same")(concat)
    # sk_conv2 = slim.conv2d(concat, 7, [5, 5], rate=1, activation_fn=lrelu)
    # sk_conv3 = slim.conv2d(concat, 7, [7, 7], rate=1, activation_fn=lrelu)
    sum_u = sk_conv1 + sk_conv2 + sk_conv3
    squeeze = GlobalAveragePooling2D()(sum_u)
    squeeze = tf.reshape(squeeze, [-1, 1, 1, out_dim])
    z = Dense(middle, use_bias=True)(squeeze)
    z = tf.nn.relu(z)
    a1 = Dense(out_dim, use_bias=True)(z)
    a2 = Dense(out_dim, use_bias=True)(z)
    a3 = Dense(out_dim, use_bias=True)(z)


    before_softmax = tf.concat([a1, a2, a3], 1)
    after_softmax = tf.nn.softmax(before_softmax)
    a1 = after_softmax[:, 0, :, :]
    a1 = tf.reshape(a1, [-1, 1, 1, out_dim])
    a2 = after_softmax[:, 1, :, :]
    a2 = tf.reshape(a2, [-1, 1, 1, out_dim])
    a3 = after_softmax[:, 2, :, :]
    a3 = tf.reshape(a3, [-1, 1, 1, out_dim])

    select_1 = sk_conv1 * a1
    select_2 = sk_conv2 * a2
    select_3 = sk_conv3 * a3

    out = select_1 + select_2 + select_3

    return out


def network(in_image):
    feature_map = feature_encoding(in_image)
    feature_map_2 = tf.concat([in_image, feature_map], 3)
    pool1, pool2, pool3, pool4, pool5 = avg_pool(feature_map_2)
    unet1, unet2, unet3, unet4, unet5 = all_unet(pool1, pool2, pool3, pool4, pool5)
    resize1, resize2, resize3, resize4, resize5 = resize_all_image(unet1, unet2, unet3, unet4, unet5)
    out_image = to_clean_image(feature_map_2, resize1, resize2, resize3, resize4, resize5)

    return out_image