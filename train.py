import datetime
import time
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from models import load_simple_fcn_with_border, load_simple_fcn_no_border, get_unet
from load_data import load_dataset


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
    from tensorflow.python.client import device_lib

    print(device_lib.list_local_devices())
    print(tf.version.VERSION)
    IMAGE_WIDTH = 128
    IMAGE_HEIGHT = 128
    BATCH_SIZE = 64
    CHANNELS = 3
    LOSS = tf.keras.losses.CategoricalCrossentropy()
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.0001)

    # DIRECTORIES & Variables
    train_path = "/CompVisData/train2"
    validation_path = "/CompVisData/val"
    test_path = "/CompVisData/test"
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    border = None

    ####### START LOAD FCN WITH BORDER ###########
    # model, border = load_simple_fcn_with_border()
    ####### END LOAD FCN WITH BORDER ###########

    ########### START LOAD FCN NO BORDER ###########
    # model = load_simple_fcn_no_border()
    ########### END LOAD FCN NO BORDER ###########

    ########### START LOAD U-NET ##########
    input_img = keras.layers.Input((IMAGE_HEIGHT, IMAGE_WIDTH, 3), name='img')
    model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
    u_net = True
    ########### END LOAD U-NET ##########

    # train, val, test =

    train_dataset = load_dataset(path=train_path, border=border, train=True)
    val_dataset = load_dataset(path=validation_path, border=border)
    test_dataset = load_dataset(path=test_path, border=border)

    train_batches = train_dataset.batch(batch_size=BATCH_SIZE)
    val_batches = val_dataset.batch(batch_size=BATCH_SIZE)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1,
                                                          write_graph=True,
                                                          write_grads=True)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                               min_delta=0.01,
                                                               patience=5,
                                                               mode='auto',
                                                               restore_best_weights=True)


    print("model loaded")
    print(model.summary())
    model.compile(optimizer='adam', loss=LOSS, metrics=["accuracy"])
    start_train_loop = int(time.time())
    # TODO: Implement early stopping
    model_history = model.fit(train_batches,
                              epochs=50,
                              validation_data=val_batches,
                              callbacks=[tensorboard_callback]
                              )


    # SAVE MODEL
    num_secs = int(time.time()) - start_train_loop
    print("Finished training")
    print(f"Model training took {str(datetime.timedelta(seconds=num_secs))}")
    visualize_history_metrics(history=model_history)


    model.save('/computer_vision/saved_models/u_net_augmented')



