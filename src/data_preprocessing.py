import pandas as pd
import numpy as np
import cv2

def preprocess_data(df):
    labels = df['label'].values
    images = df.drop('label', axis=1).values.reshape(-1, 28, 28, 1)
    images = images.astype(np.float32) / 255.0
    resized_images = np.array([cv2.resize(img.squeeze(), (64, 64)) for img in images]).reshape(-1, 64, 64, 1)
    return resized_images, labels

train_df = pd.read_csv('../data/sign_mnist_train.csv')
X_train, y_train = preprocess_data(train_df)
np.save('../data/X_train.npy', X_train)
np.save('../data/y_train.npy', y_train)

test_df = pd.read_csv('../data/sign_mnist_test.csv')
X_test, y_test = preprocess_data(test_df)
np.save('../data/X_test.npy', X_test)
np.save('../data/y_test.npy', y_test)