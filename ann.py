import keras
from keras import (
    layers,
    activations,
    initializers,
    regularizers,
    optimizers,
    losses,
    metrics,
)
import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

X_useful = np.array()
y_useful = np.array()

sss = StratifiedShuffleSplit(n_splits=1, random_state=32, test_size=0.3)
for tr, te in sss.split(X_useful, y_useful):
    X_train, y_train = X_useful.iloc[tr], y_useful.iloc[tr]
    X_test, y_test = X_useful.iloc[te], y_useful.iloc[te]

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))


def build_model(input_shape, n_layers, units, n_classes, drop_rate):
    input_layer = layers.Input(shape=input_shape)

    x = input_layer
    for _ in range(n_layers):
        if units < 25:
            units = 25
        x = layers.Dense(
            units,
            activation=activations.leaky_relu,
            kernel_initializer=initializers.he_normal,
            kernel_regularizer=regularizers.l2(0.01),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(rate=drop_rate)(x)
        units //= 2

    if n_classes > 2:
        activation = activations.softmax
    else:
        activation = activations.sigmoid
    output_layer = layers.Dense(
        n_classes, activation=activation, kernel_initializer=initializers.GlorotNormal
    )(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model


model = build_model()
opt = optimizers.Adam(learning_rate=0.001)
loss = losses.BinaryCrossentropy
metric = [metrics.binary_accuracy]
model.compile(optimizer=opt, loss=loss, metrics=metric)


import optuna

input_shape = (299,)
train = tf.data.Dataset


import optuna


def objective(trial):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        n_layers = trial.suggest_int("n_layers", 2, 10)
        units = trial.suggest_int("units", 100, 1000)
        drop_out_rate = trial.suggest_float("drop_out_rate", 0.0, 0.6)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-7, 1e-2)
        shuffle_size = trial.suggest_int("shuffle_size", 100, 5000)
        batch_size = trial.suggest_int("batch_size", 16, 256)
        epochs = trial.suggest_int("epochs", 10, 1000)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        train = (
            train_dataset.shuffle(shuffle_size)
            .batch(batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        val = (
            val_dataset.shuffle(shuffle_size)
            .batch(batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        model = build_model(
            input_shape=input_shape,
            n_layers=n_layers,
            units=units,
            n_classes=1,
            drop_rate=drop_out_rate,
        )

        opt = optimizers.Adam(learning_rate=learning_rate)
        loss = losses.BinaryCrossentropy
        metric = [metrics.binary_accuracy]
        model.compile(optimizer=opt, loss=loss, metrics=metric)

        history = model.fit(train, epochs=epochs, validation_data=val)
        val_loss = history["val_loss"]
        train_loss = history["train_loss"]
    return train_loss, val_loss


study = optuna.create_study(directions=["minimize", "minimize"])
study.optimize(objective, n_trials=5)
