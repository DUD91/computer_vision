import os
import datetime
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from models import load_simple_fcn_with_border, load_simple_fcn_no_border


def _parse_function(image_filename: str,
                    label_filename: str,
                    channels: int,
                    border: int = None):
    image_string = tf.io.read_file(image_filename)
    image_decoded = tf.image.decode_png(image_string, channels=channels)
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
    if border is not None:
        label = label[border:-border, border:-border]

    # ONE-HOT ENCODING (casting due to 2.5 in labels)
    label = tf.dtypes.cast(label, tf.int32)
    label = tf.one_hot(label, NR_CLASSES)
    label = tf.reshape(label, shape=[label.shape[0], label.shape[1], 34])

    return image, label


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


def load_dataset(path: str, border:int =None) -> tf.data.Dataset:
    files = load_filenames(path)
    images, labels = get_images_and_labels(files)
    images.sort()
    labels.sort()

    # CREATE DATASET
    image_files_array = np.asarray([str(p) for p in images])
    label_files_array = np.asarray([str(p) for p in labels])

    dataset = tf.data.Dataset.from_tensor_slices((image_files_array, label_files_array))

    # # shuffle the filename, unfortunately, then we cannot cache them
    dataset = dataset.shuffle(buffer_size=len(images))
    # # read the images
    dataset = dataset.map(lambda image, label: _parse_function(image_filename=image,
                                                               label_filename=label,
                                                               channels=CHANNELS,
                                                               border=border,
                                                               ))

    return dataset


if __name__ == "__main__":
    from tensorflow.python.client import device_lib

    print(device_lib.list_local_devices())
    IMAGE_WIDTH = 128
    IMAGE_HEIGHT = 128
    BATCH_SIZE = 64
    CHANNELS = 3
    LOSS = tf.keras.losses.CategoricalCrossentropy()
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.0001)
    NR_CLASSES = 34
    border = None


    # DIRECTORIES
    train_path = "/Users/dani/repositories/computer_vision/CompVisData/train"
    validation_path = "/Users/dani/repositories/computer_vision/CompVisData/val"
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Load FCN with border
    # model, border = load_simple_fcn_with_border()

    # Load FCN no border
    # model = load_simple_fcn_no_border()

    # Load U-NET
    input_img = keras.layers.Input((IMAGE_HEIGHT, IMAGE_WIDTH, 3), name='img')
    model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
    u_net = True

    train_dataset = load_dataset(path=train_path, border=border, u_net=u_net)
    val_dataset = load_dataset(path=validation_path, border=border, u_net=u_net)

    # NOT WORKING ON GPU M1
    # data_augmentation = tf.keras.Sequential([
    #     layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    #     layers.experimental.preprocessing.RandomRotation(0.2),
    # ])
    # train_dataset_aug = train_dataset.map(lambda x, y: (data_augmentation(x), y))
    # train_batches = train_dataset_aug.batch(batch_size=BATCH_SIZE)

    train_batches = train_dataset.batch(batch_size=BATCH_SIZE)
    val_batches = val_dataset.batch(batch_size=BATCH_SIZE)

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
