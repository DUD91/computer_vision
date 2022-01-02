import os
import numpy as np
import tensorflow as tf
from typing import List

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
CHANNELS = 3
NR_CLASSES = 34

def _parse_function(image_filename: str,
                    label_filename: str,
                    border: int = None):

    image_string = tf.io.read_file(image_filename)
    image_decoded = tf.image.decode_png(image_string, channels=CHANNELS)
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
    label = tf.reshape(label, shape=[label.shape[0], label.shape[1], NR_CLASSES])

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


def augment_data(image, label_mask):
    # Randomly flip input images and label for robustness
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        label_mask = tf.image.flip_left_right(label_mask)
    return image, label_mask


def load_dataset(path: str, border:int = None, train: bool = False) -> tf.data.Dataset:
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
                                                               border=border,
                                                               ))

    if train:
        dataset = dataset.map(lambda image, label: augment_data(image=image, label_mask=label))

    return dataset


def get_train_val_test_split(data_dir: str, border: int = None, dryrun: bool = False):
    train_path = data_dir + "train"
    if not dryrun:
        train_path += "2"
    validation_path = data_dir + "val"
    test_path = data_dir + "test"

    train_dataset = load_dataset(path=train_path, border=border, train=True)
    val_dataset = load_dataset(path=validation_path, border=border)
    test_dataset = load_dataset(path=test_path, border=border)

    return train_dataset, val_dataset, test_dataset


