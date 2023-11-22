import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb
import tensorflow as tf


(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(
    num_words=10000
)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)
print(data.shape)


def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


data = vectorize(data)
targets = np.array(targets).astype("float32")
test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]

model = models.Sequential()

# initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
initializer = tf.keras.initializers.GlorotUniform()

# Input - Layer
model.add(
    layers.Dense(
        50, activation="relu", input_shape=(10000,), kernel_initializer=initializer
    )
)
# Hidden - Layers
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation="relu", kernel_initializer=initializer))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation="relu", kernel_initializer=initializer))
# model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
# model.add(layers.Dense(50, activation="relu", kernel_initializer=initializer))

# Output- Layer
model.add(layers.Dense(1, activation="sigmoid"))
model.summary()

# compiling the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# tensorboard
log_dir = "logs/fit/hidden3_epochs20"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


results = model.fit(
    train_x,
    train_y,
    epochs=20,
    batch_size=500,
    validation_data=(test_x, test_y),
    callbacks=[tensorboard_callback],
)

print("Test-Accuracy:", np.mean(results.history["val_accuracy"]))
