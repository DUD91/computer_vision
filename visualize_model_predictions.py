import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from load_data import get_train_val_test_split


CHANNELS = 3
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
border = None

trained_model = tf.keras.models.load_model('saved_models/u_net')


def create_mask(mask):
    mask = tf.argmax(mask, axis=-1)
    return mask


def show_predictions(model, dataset=None,  num=1):
    if dataset:
        for image, mask in dataset.take(num):
            mask = create_mask(mask)
            image = tf.reshape(image, shape=[1, image.shape[0], image.shape[1], image.shape[2]])
            pred_mask = model.predict(image)
            pred_mask = create_mask(pred_mask)
            display_sample([image[0], mask, pred_mask[0]])


def display_sample(display_list):
    """Show side-by-side an input image,
    the ground truth and the prediction.
    """
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        shape = display_list[i].shape
        if len(shape) > 2:
            image = tf.keras.preprocessing.image.array_to_img(display_list[i])
            plt.imshow(image, vmin=1, vmax=255)
        else:
            print(np.unique(display_list[i]))
            plt.imshow(display_list[i], vmin=0, vmax=34)
        plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # Check its architecture
    trained_model = tf.keras.models.load_model('saved_models/u_net_augmented_e50')
    trained_model.summary()

    data_dir = "./CompVisData/"

    # Check model summary (output-layer) if border is not None (only applicable for FCN with border)
    border = None
    _, _, test_dataset = get_train_val_test_split(data_dir=data_dir, border=border)

    show_predictions(dataset=test_dataset, model=trained_model, num=20)






