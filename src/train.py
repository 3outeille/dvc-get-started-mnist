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

    whole_train_ds, whole_train_labels = load_npz_data("data/preprocessed/mnist-train.npz")
    validation_split_index = int((1 - params["validation_split"]) * whole_train_ds.shape[0])
    X_train = whole_train_ds[:validation_split_index]
    X_valid = whole_train_ds[validation_split_index:]
    y_train = whole_train_labels[:validation_split_index]
    y_valid = whole_train_labels[validation_split_index:]

    m.fit(X_train, y_train,
              batch_size = params["batch_size"],
              epochs = params["epochs"],
              verbose=1,
              validation_data = (X_valid, y_valid))

    m.save("models/model.h5")


if __name__ == "__main__":
    main()
