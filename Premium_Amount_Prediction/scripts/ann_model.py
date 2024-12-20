from xgboost_training import final_model, X_train, X_test, y_train, y_test
import tensorflow as tf
import keras
from keras import layers, initializers, regularizers, activations, Model, losses, metrics

y_train_pred = final_model.predict(X_train)
train_residual_errors = abs(y_train - y_train_pred)

y_test_pred = final_model.predict(X_test)
test_residual_errors = abs(y_test - y_test_pred)

train_errors_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, train_residual_errors)
)
test_errors_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test_pred))

input_layer = layers.Input(shape=(X_train.shape[1],))
x = input_layer
units = 520
for _ in range(3):
    x = layers.Dense(
        units,
        kernel_initializer=initializers.HeUniform,
        kernel_regularizer=regularizers.L1L2,
        activation=activations.gelu,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(rate=0.5)(x)
output_layer = layers.Dense(1, activation=activations.linear)(x)

ann_model = Model(inputs=input_layer, outputs=output_layer)

print(ann_model.summary())

ann_model.compile(optimizer='adam', loss=losses.MeanSquaredError, metrics=metrics.MeanSquaredError)
BATHCH_SIZE = 512 * 2
train_errors_dataset_batched = train_errors_dataset.shuffle(1000).batch(BATHCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
ann_model.fit(train_errors_dataset, epochs=50, shuffle=True, validation_split=0.2)

losses.MeanSquaredLogarithmicError