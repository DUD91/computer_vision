import os
import datetime
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt




def _parse_function(image_filename: str, label_filename: str, channels: int, border: int):
    image_string = tf.io.read_file(image_filename)
    image_decoded = tf.image.decode_png(image_string, channels=channels)
    #image = tf.cast(image_decoded, tf.float32) / 255.0
    image = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)

    # normalize image to zero mean
    image = tf.multiply(image, 2.0)
    image = tf.subtract(image, 1.0)

    # Resize the image to target size
    image = tf.image.resize(image, size=(IMAGE_HEIGHT, IMAGE_WIDTH))

    label_string = tf.io.read_file(label_filename)
    label = tf.image.decode_png(label_string)
    # Resize the label to target size
    label = tf.image.resize(label, size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    label = label[border:-border, border:-border]
    label = tf.dtypes.cast(label, tf.int32)
    label = tf.one_hot(label, NR_CLASSES)
    label = tf.reshape(label, shape=[label.shape[0],label.shape[1],34])

    #label = keras.utils.to_categorical(label, num_classes=NR_CLASSES)

    return image, label


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

    # go to logits which is the number of classes and add sigmoid layer for activation
    model.add(keras.layers.Convolution2D(filters=4, kernel_size=1, padding="valid", activation="relu"))

    model.add(keras.layers.Convolution2D(filters=34, kernel_size=1, activation=None,
                                         kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001, seed=None)))

    model.add(keras.layers.Activation('softmax'))
    #model.add(keras.layers.Reshape(target_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)))

    return model, border


def get_images_and_labels(images):
    image_files = []
    label_files = []
    for filename in images:
        if "_img" in str(filename):
            image_files.append(filename)
        else:
            label_files.append(filename)
    return image_files, label_files

def load_filenames(path):
    for file in os.listdir(path):
        if file.endswith(".png"):
            filepath = os.path.join(path, file)
            yield filepath

def display_sample(display_list):
    """Show side-by-side an input image,
    the ground truth and the prediction.
    """
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def load_dataset(path: str, border: int) -> tf.data.Dataset:
    files = load_filenames(path)
    images, labels = get_images_and_labels(files)
    images.sort()
    labels.sort()
    #CREATE DATASET
    image_files_array = np.asarray([str(p) for p in images])
    label_files_array = np.asarray([str(p) for p in labels])

    dataset = tf.data.Dataset.from_tensor_slices((image_files_array, label_files_array))

    # # shuffle the filename, unfortunately, then we cannot cache them
    dataset = dataset.shuffle(buffer_size=len(images))
    # # read the images
    dataset = dataset.map(lambda image, label: _parse_function(image_filename=image, label_filename=label, channels=CHANNELS, border = border))

    return dataset



if __name__ == "__main__":
    from tensorflow.python.client import device_lib

    print(device_lib.list_local_devices())
    IMAGE_WIDTH = 128
    IMAGE_HEIGHT = 128
    BATCH_SIZE = 64
    CHANNELS = 3
    #OPTIMIZER = tf.keras.optimizers.SGD(momentum=0.9)
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.0001)
    LOSS = tf.keras.losses.CategoricalCrossentropy()
    #LOSS = tf.keras.losses.Spar()
    NR_CLASSES = 34

    train_path = "/Users/dani/repositories/computer-vision/assignment02/CompVisData/train"
    validation_path = "/Users/dani/repositories/computer-vision/assignment02/CompVisData/val"

    # Load model
    model, border = load_simple_fcn_with_border()

    train_dataset = load_dataset(path=train_path, border = border)
    val_dataset = load_dataset(path=validation_path, border = border)

    # data_augmentation = tf.keras.Sequential([
    #     layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    #     layers.experimental.preprocessing.RandomRotation(0.2),
    # ])

    # train_dataset_aug = train_dataset.map(lambda x, y: (data_augmentation(x), y))

    train_batches = train_dataset.batch(batch_size=BATCH_SIZE)
    #train_batches = train_dataset_aug.batch(batch_size=BATCH_SIZE)

    val_batches = val_dataset.batch(batch_size=BATCH_SIZE)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1,
                                                          write_graph=True,
                                                          write_grads=True)

    print("model loaded")
    print(model.summary())
    model.compile(optimizer='adam', loss=LOSS, metrics=["accuracy"])
    history = model.fit(train_batches,
                        epochs=30,
                        validation_data=val_batches,
                        callbacks=[tensorboard_callback]
                        )
