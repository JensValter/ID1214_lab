import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


#task 1
# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = (
    keras.datasets.mnist.load_data()
)

print("Training images:", X_train.shape)
print("Training labels:", y_train.shape)
print("Test images:", X_test.shape)
print("Test labels:", y_test.shape)


print("Pixel values:", X_train.min(), X_train.max())
print("Example label:", y_train[0])


#task 2
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
print("Normalised Pixel values X_train:", X_train.min(), X_train.max())
print("Normalised Pixel values X_test:", X_test.min(), X_test.max())

#task 3
model = keras.Sequential([
    keras.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.summary()

#task 4
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)

#task 5
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

training_loss, training_accuracy = model.evaluate(X_train, y_train)
print("Training loss:", training_loss)
print("Training accuracy:", training_accuracy)


#hidden layer with 256 neurons
print("")
print("hidden layer with 256 neurons")

model = keras.Sequential([
    keras.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

training_loss, training_accuracy = model.evaluate(X_train, y_train)
print("Training loss:", training_loss)
print("Training accuracy:", training_accuracy)


#7 epochs
print("")
print("7 epochs")
model = keras.Sequential([
    keras.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train,
    epochs=7,
    batch_size=128,
    validation_split=0.1
)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

training_loss, training_accuracy = model.evaluate(X_train, y_train)
print("Training loss:", training_loss)
print("Training accuracy:", training_accuracy)


# another hidden layer
print("")
print("another hidden layer")
model = keras.Sequential([
    keras.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

training_loss, training_accuracy = model.evaluate(X_train, y_train)
print("Training loss:", training_loss)
print("Training accuracy:", training_accuracy)