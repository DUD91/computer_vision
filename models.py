import tensorflow.keras as keras


IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
NR_CLASSES = 34


def load_simple_fcn_with_border():
    border = 0
    model = keras.Sequential()

    # (used to define input shape on the first layers)
    model.add(keras.layers.Layer(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))

    # add 3 convolutional layers with 3x3 filters
    model.add(keras.layers.Convolution2D(filters=4, kernel_size=3, padding='valid', activation='relu'))
    border += 1
    model.add(keras.layers.Convolution2D(filters=8, kernel_size=3, padding='valid', activation='relu'))
    border += 1
    model.add(keras.layers.Convolution2D(filters=16, kernel_size=3, padding='valid', activation='relu'))
    border += 1
    model.add(keras.layers.Convolution2D(filters=32, kernel_size=3, padding='valid', activation='relu'))
    border += 1
    model.add(keras.layers.Convolution2D(filters=64, kernel_size=3, padding='valid', activation='relu'))
    border += 1
    model.add(keras.layers.Convolution2D(filters=32, kernel_size=3, padding='valid', activation='relu'))
    border += 1
    model.add(keras.layers.Convolution2D(filters=16, kernel_size=3, padding='valid', activation='relu'))
    border += 1
    model.add(keras.layers.Convolution2D(filters=8, kernel_size=3, padding='valid', activation='relu'))
    border += 1
    model.add(keras.layers.Convolution2D(filters=4, kernel_size=1, padding="valid", activation="relu"))
    # Output equals to Nr. of classes
    model.add(keras.layers.Convolution2D(filters=NR_CLASSES, kernel_size=1, activation=None,
                                         kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001, seed=None)))

    model.add(keras.layers.Activation('softmax'))

    return model, border


def load_simple_fcn_no_border():
    model = keras.Sequential()

    # (used to define input shape on the first layers)
    model.add(keras.layers.Layer(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))

    # add 3 convolutional layers with 3x3 filters
    model.add(keras.layers.Convolution2D(filters=4, kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.Convolution2D(filters=8, kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.Convolution2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.Convolution2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.Convolution2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.Convolution2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.Convolution2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.Convolution2D(filters=8, kernel_size=3, padding='same', activation='relu'))

    # go to logits which is the number of classes and add sigmoid layer for activation
    model.add(keras.layers.Convolution2D(filters=4, kernel_size=1, padding="same", activation="relu"))

    # Output equals to Nr. of classes
    model.add(keras.layers.Convolution2D(filters=NR_CLASSES, kernel_size=1, activation=None,
                                         kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0,
                                                                                               stddev=0.001,
                                                                                               seed=None)))

    model.add(keras.layers.Activation('softmax'))
    return model


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = keras.layers.Convolution2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
               kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # second layer
    x = keras.layers.Convolution2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
               kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    return x


def get_unet(input_img, n_filters=16, dropout=0.1, batchnorm=True):
    """Function to define the UNET Model"""
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)
    p1 = keras.layers.Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)
    p2 = keras.layers.Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)
    p3 = keras.layers.Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = keras.layers.MaxPooling2D((2, 2))(c4)
    p4 = keras.layers.Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # Expansive Path
    u6 = keras.layers.Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = keras.layers.concatenate([u6, c4])
    u6 = keras.layers.Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = keras.layers.Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = keras.layers.concatenate([u7, c3])
    u7 = keras.layers.Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = keras.layers.Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = keras.layers.concatenate([u8, c2])
    u8 = keras.layers.Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = keras.layers.Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = keras.layers.concatenate([u9, c1])
    u9 = keras.layers.Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = keras.layers.Conv2D(NR_CLASSES, (1, 1), activation='sigmoid')(c9)
    model = keras.Model(inputs=[input_img], outputs=[outputs])
    return model

