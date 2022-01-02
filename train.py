import os
import datetime
import time
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from models import load_simple_fcn_with_border, load_simple_fcn_no_border, get_unet
from load_data import get_train_val_test_split


def visualize_history_metrics(history):
    loss, acc = history.history['loss'], history.history['accuracy']
    val_loss, val_acc = history.history['val_loss'], history.history['val_accuracy']
    fig, axs = plt.subplots(nrows=2, figsize=(15, 10))

    # LOSS
    axs[0].plot(history.epoch, loss, 'r', label='Training loss')
    axs[0].plot(history.epoch, val_loss, 'bo', label='Validation loss')
    axs[0].set(xlabel="Epoch", ylabel="Loss Value", title='Training and Validation Loss')
    axs[0].legend()

    # ACCURACY
    axs[1].plot(history.epoch, acc, 'r', label='Training Accuracy')
    axs[1].plot(history.epoch, val_acc, 'bo', label='Validation Accuracy')
    axs[1].set(xlabel="Epoch", ylabel="Accuracy Value", title='Training and Validation Accuracy')
    axs[1].legend()

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    IMAGE_WIDTH = 128
    IMAGE_HEIGHT = 128
    BATCH_SIZE = 64
    CHANNELS = 3
    LOSS = tf.keras.losses.CategoricalCrossentropy()
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.0001)

    # DIRECTORIES & Variables
    data_dir = "./CompVisData/"
    log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    input = input("Insert one of the following options: \n 1: FCN without border \n 2: FCN with border \n 3: u_net \n")
    print("Chosen input:", input)

    border = None
    if input == 1:
        model = load_simple_fcn_no_border()
    elif input == 2:
        model, border = load_simple_fcn_with_border()
    else:
        input_img = keras.layers.Input((IMAGE_HEIGHT, IMAGE_WIDTH, 3), name='img')
        model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

    train_dataset, val_dataset, test_dataset = get_train_val_test_split(data_dir=data_dir, border=border)

    # Create dataset batches
    train_batches = train_dataset.batch(batch_size=BATCH_SIZE)
    val_batches = val_dataset.batch(batch_size=BATCH_SIZE)

    # Define Callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1,
                                                          write_graph=True,
                                                          write_grads=True)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                               min_delta=0.01,
                                                               patience=5,
                                                               mode='auto',
                                                               restore_best_weights=True)

    print("Model loaded")
    print(model.summary())
    epochs = 100
    model.compile(optimizer='adam', loss=LOSS, metrics=["accuracy"])
    start_train_loop = int(time.time())
    model_history = model.fit(train_batches,
                              epochs=epochs,
                              validation_data=val_batches,
                              callbacks=[tensorboard_callback, early_stopping_callback]
                              )

    # SAVE & Visualize MODEL
    num_secs = int(time.time()) - start_train_loop
    print("Finished training")
    print(f"Model training took {str(datetime.timedelta(seconds=num_secs))}")
    visualize_history_metrics(history=model_history)

    model.save('./computer_vision/saved_models/u_net_augmented_e100')



