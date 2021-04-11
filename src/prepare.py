import struct
import numpy as np
import gzip
import os

def mnist_images_idx_to_array(images_filename):
    images_f = gzip.open(images_filename, mode="rb")
    images_f.seek(0)
    magic = struct.unpack('>I', images_f.read(4))[0]
    if magic != 0x00000803:
        raise Exception(f"Format error: Need an IDX3 file: {images_filename}")
    n_images = struct.unpack('>I', images_f.read(4))[0]
    n_row = struct.unpack('>I', images_f.read(4))[0]
    n_col = struct.unpack('>I', images_f.read(4))[0]

    n_bytes = n_images * n_row * n_col # each pixel is 1 byte

    images_data = struct.unpack('>' + str(n_bytes) + 'B', images_f.read(n_bytes))

    images_array = np.asarray(images_data, dtype='uint8')
    images_array.shape = (n_images, n_row, n_col)

    return images_array

def mnist_labels_idx_to_array(labels_filename):
    labels_f = gzip.open(labels_filename, mode="rb")
    labels_f.seek(0)
    magic = struct.unpack('>I', labels_f.read(4))[0]
    if magic != 0x00000801:
        raise Exception(f"Format error: Need an IDX file: {labels_filename}")
    n_labels = struct.unpack('>I', labels_f.read(4))[0]
    labels_data = struct.unpack('>' + str(n_labels) + 'B', labels_f.read(n_labels))
    labels_array = np.asarray(labels_data, dtype='uint8')
    return labels_array


def main():
    training_images = mnist_images_idx_to_array("data/train-images-idx3-ubyte.gz")
    print(f"Read training data: {training_images}")
    training_labels = mnist_labels_idx_to_array("data/train-labels-idx1-ubyte.gz")
    print(f"Read training labels: {training_labels}")
    testing_images = mnist_images_idx_to_array("data/t10k-images-idx3-ubyte.gz")
    print(f"Read testing data: {testing_images}")
    testing_labels = mnist_labels_idx_to_array("data/t10k-labels-idx1-ubyte.gz")
    print(f"Read testing labels: {testing_labels}")

    np.savez("data/mnist-train.npz", images=training_images, labels=training_labels)
    np.savez("data/mnist-test.npz", images=testing_images, labels=testing_labels)


if __name__ == "__main__":
    main()
