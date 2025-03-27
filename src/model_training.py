from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import mlflow
import numpy as np

X_train = np.load('../data/X_train.npy')
y_train = np.load('../data/y_train.npy')
X_test = np.load('../data/X_test.npy')
y_test = np.load('../data/y_test.npy')

y_train = to_categorical(y_train, 25)
y_test = to_categorical(y_test, 25)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(25, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

with mlflow.start_run():
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    mlflow.log_metric("test_accuracy", model.evaluate(X_test, y_test)[1])
    mlflow.tensorflow.log_model(model, "model")

model.save('../models/gesture_model.h5')