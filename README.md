# DVC Example Get Started (MNIST)

This repository contains the new (2021) example get-started project. It employs
Tensorflow 2.4 with the standard [MNIST][mnist] dataset. The dataset is
downloaded not from the TF datasets but from the [DVC Dataset Registry][dsr], all preprocessing code
is custom to this project and Tensorflow dependency is kept minimal. This project is used as a showcase for the experimentation features
in DVC 2.0.

[mnist]: http://yann.lecun.com/exdb/mnist/
[dsr]: https://github.com/iterative/dataset-registry

## Installation

After cloning the project, you can create a virtual environment and activate it:

```console
python3 -m venv .env 
source .env/bin/activate
```

Install the requirements.

```console
pip3 install -r requirements.txt
```

Run the pipeline:

```console
dvc exp run
```

## Parameters

This project is aimed towards experimentation and thus contains many parameters
to change and play. All of these are set in `params.yaml` and can be modified during
experiments as:

```console
dvc exp run --set-param model.name=mlp \
            --set-param model.mlp.units=128 \
            --set-param model.mlp.activation=relu
```

### Parameters for the `prepare` stage

- `prepare.remix`: Determines whether MNIST train (60000 images) and
  test (10000 images) sets are merged and split. If `false`, MNIST test and
  train sets are not merged and used as in the original.

- `prepare.remix_split`: Determines the split ratio between training and testing
  sets if `remix` is `true`. For `0.20`, a total of 70000 images are randomly
  split into 56000 training and 14000 test sets.

- `prepare.seed`: The RNG seed used in shuffling after the remix.

### Parameters for the `preprocess` stage

- `preprocess.seed`: The RNG seed used in shuffling. 

- `preprocess.normalize`: If `true`, normalizes the pixel values (0-255) dividing by 255.
  Although this is a standard and required procedure, you may want to observe
  the effects by turning it off.

- `preprocess.shuffle`: If `true`, shuffles the training and test sets. 

- `preprocess.add_noise`: If `true` adds salt-and-pepper noise by setting some pixels to
  white and some pixels to black. This may be used to reduce overfitting.
  
- `preprocess.noise_amount`: Sets the amount of S&P noise added to the images if
  `add_noise` is `true`.
  
- `preprocess.noise_s_vs_p`: Sets the ratio of white and black noise in images if
  `add_noise` is `true`.

### Parameters for the `train` stage

- `train.validation_split`: The split ratio for the validation set, reserved from the
  training set. If this value is `0`, the test set is used for validation. 

- `train.epochs`: Number of epochs to train the network. 

- `train.batch_size`: Batch size for the `model.fit` method. 

### Parameters for the `model`

These parameters are used to set the attributes of the models. Although their
structure is fixed, you can set some important parameters that will affect the
performance, like `units` for the MLP or `conv_units` for the CNN.

- `model.name`: Used to select the model. For `mlp` a simple NN with a single
  hidden layer is used. For `cnn`, a Convolutional Net with a single `Conv2D`
  and a single `Dense` layer is used. The parameters for these networks are defined in separate sections below.
  
- `model.optimizer`: Can be one of `Adam`, `SGD`, `RMSprop`, `Adadelta`, `Adagrad`, `Adamax`, `Nadam`, `Ftrl`.

- `model.mlp.units`: Number of `Dense` units in MLP.

- `model.mlp.activation`: Activation function for the `Dense` layer. Can be one of `relu`, `selu`, `elu`, `tanh`

- `model.cnn.dense_units`: Number of units in `Dense` layer of the CNN.

- `model.cnn.activation`: The activation function for the convolutional layer.
  Can be one of `relu`, `selu`, `elu` or `tanh`.

- `model.cnn.conv_kernel_size`: One side of convolutional kernel, e.g., for `3`, a `(3, 3)` convolution applied to the images.

- `model.cnn.conv_units`: Number of convolutional units. 

- `model.cnn.dropout`: Dropout rate between `0` and `1`. Usually set to `0.5`.

### Selecting metrics to report

These parameters are used to select the appropriate metrics to generate during
training and evaluation. These also affect the columns/fields of `train.log.csv`
and `metrics.json` files.

- `model.metrics.categorical_accuracy`: Produces accuracy metrics for the classes.

- `model.metrics.recall`: Recall metric (True Positives / All Relevant Elements)

- `model.metrics.precision`: Precision metric (True Positives / All Positives)

- `model.metrics.auc-roc`: Generates [Receiver Operating
  Characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
  curve

- `model.metrics.auc-prc`: Generates Precision-Recall Curve

- `model.metrics.fp`: Number of False Positives

- `model.metrics.fn`: Number of False Negatives

- `model.metrics.tp`: Number of True Positives

- `model.metrics.tn`: Number of True Negatives

## Files

### Data Files

The data files used in the project are found in `data/`. All of these files are
tracked by DVC and can be retrieved using `dvc pull` from the configured remote.

- `data/raw.dvc`: Contains a reference to the [Dataset
  Registry][dsr] to download the MNIST
  dataset to `data/raw/`.

- `data/prepared/`: Created by `src/prepare.py` and contains training and testing files in NumPy format.

- `data/preprocessed/`: Created by `src/preprocess.py` and contains training and
  testing files in NumPy format ready to be supplied to `model.train`.

### Source Files

The source files are `src/` directory. All files receive runtime parameters from
`params.yaml`, so none of them require any options. File dependencies are
hardcoded in the current version, but this may change in a later iteration.
Almost all capabilities of these scripts can be modified with the options in `params.yaml`

- `src/prepare.py`: Reads the raw dataset files from `data/raw/` in _IDX3_ format and converts to
  NumPy format. As the MNIST dataset already contains train and test sets, this script
  can remix and split them if needed. The output files are stored in
  `data/prepared/`.
  
- `src/preprocess.py`: Reads data files from `data/prepared/` and adds salt and
  pepper noise, normalize the values and shuffles. The output in
  `data/preprocessed/` is ready to supply to the Neural Network.

- `src/models.py`: Contains two models. The first one is an MLP with a single
  hidden layer.  The second is a deeper network with a convolution layer, max
  pooling, dropout, and a hidden dense layer. Various parameters of these
  networks can be set in `params.yaml`. The metrics produced as the output are
  also compiled into models in this file. The metrics can be turned on-and-off
  in `params.yaml` as described above.

- `src/train.py`: Trains the specific neural network returned by `src/models.py` with the
  data in `data/preprocessed/`. It produces 
  `train.log.csv` plots file during the training that contains various metrics
  for each epoch, and `models/model.h5` file at the end. 

- `src/evaluate.py`: Tests the model `models/model.h5` created by the training
  stage, with the test data in `data/preprocessed`. It produces `metrics.json`
  file that has the testing metrics of the model.

- `requirements.txt`: Contains the requirements to run the project.
  
### Model Files

- `models/model.h5`: The Tensorflow model produced by `src/train.py` in HDF5
  format.

### Metrics and Plots

Following two files are tracked by DVC as plots and metrics files, respectively.


- `train.log.csv`: Training and validation metrics in each epoch produced in
  `src/train.py` is written to this file.

- `metrics.json`: Final metrics produced by the test set is output to this file.
  

## DVC Files

The repository is a standard Git repository and contains the usual `.dvc` files:

- `.dvc/config`: Contains a remote configuration to retrieve dataset from S3.
- `dvc.yaml`: Contains the pipeline configuration.
- `dvc.lock`: Parameters and dependency hashes are tracked with this file.

## The Pipeline

The pipeline graph retrieved by `dvc dag` is as shown below:

```
+--------------+ 
| data/raw.dvc | 
+--------------+ 
        *        
        *        
        *        
  +---------+    
  | prepare |    
  +---------+    
        *        
        *        
        *        
 +------------+  
 | preprocess |  
 +------------+  
        *        
        *        
        *        
    +-------+    
    | train |    
    +-------+    
        *        
        *        
        *        
  +----------+   
  | evaluate |   
  +----------+   
```

The following graph shows the data and output dependencies retrieved with `dvc
dag -o`

```
                  +----------+                    
                  | data/raw |                    
                  +----------+                    
                        *                         
                        *                         
                        *                         
               +---------------+                  
               | data/prepared |                  
               +---------------+                  
                        *                         
                        *                         
                        *                         
             +-------------------+                
             | data/preprocessed |                
             +-------------------+                
               ***            ***                 
             **                  **               
           **                      **             
+---------------+            +-----------------+  
| train.log.csv |            | models/model.h5 |  
+---------------+            +-----------------+  
                                      *           
                                      *           
                                      *           
                              +--------------+    
                              | metrics.json |    
                              +--------------+    
```