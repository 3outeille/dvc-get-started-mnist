import tensorflow as tf
import numpy as np
import yaml

import model

def load_npz_data(filename):
    npzfile = np.load(filename)
    return (npzfile['images'], npzfile['labels'])

def load_params():
    return yaml.safe_load(open("params.yaml"))["train"]

def main():
    params = load_params()
    m = model.get_model()
    m.summary()

    whole_train_img, whole_train_labels = load_npz_data("data/preprocessed/mnist-train.npz")
    test_img, test_labels = load_npz_data("data/preprocessed/mnist-train.npz")
    validation_split_index = int((1 - params["validation_split"]) * whole_train_img.shape[0])
    if validation_split_index == whole_train_img.shape[0]:
        x_train = whole_train_img
        x_valid = test_img
        y_train = whole_train_labels
        y_valid = test_labels
    else:
        x_train = whole_train_img[:validation_split_index]
        x_valid = whole_train_img[validation_split_index:]
        y_train = whole_train_labels[:validation_split_index]
        y_valid = whole_train_labels[validation_split_index:]

    print(f"x_train: {x_train.shape}")
    print(f"x_valid: {x_valid.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_valid: {y_valid.shape}")

    m.fit(x_train, y_train,
              batch_size = params["batch_size"],
              epochs = params["epochs"],
              verbose=1,
              validation_data = (x_valid, y_valid))

    m.save("models/model.h5")


if __name__ == "__main__":
    main()
