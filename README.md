# DVC Example Get Started (MNIST)

This repository contains the new (2021) example get started project. Instead of
the sklearn based Random Forest classification using a minimized Stack Overflow
tagging dataset used [in 
the previous](https://github.com/iterative/example-get-started) one, it uses
Tensorflow 2.4 with the standard [MNIST] dataset. This project is planned to be
used to show the experimentation features in DVC 2.0.

## Installation



## Parameters

This project is aimed towards experimentation and thus contains many parameters
to modify. All of these are set in `params.yaml` and can be modified during
experiments as:

```console
dvc exp run --set-param model.name=mlp \
            --set-param model.mlp.units=128 \
            --set-param model.mlp.activation=relu
```

### Parameters for the `prepare` stage

- `remix`: Determines whether MNIST train (60000 images) and
  test (10000 images) sets are merged and split. If `false`, MNIST test and
  train sets are not merged and used as in the original.
- `remix_split`: Determines the split ratio between training and testing
  sets if `remix` is `true`. For `0.20`, a total of 70000 images are randomly
  split into 56000 training and 14000 test sets.
- `seed`: The RNG seed used in shuffling after the remix.

### Parameters for the `preprocess` stage

- `seed`: The RNG seed used in shuffling. 
- `normalize`: If `true`, normalizes the pixel values (0-255) dividing by 255.
  Although this is a standard and required procedure, you may want to observe
  the effects by turning it off.
- `shuffle`: If `true`, shuffles the training and test sets. 
- `add_noise`: If `true` adds salt-and-pepper noise by setting some pixels to
  white and some pixels to black. This may be used to reduce overfitting.
- `noise_amount`: Sets the amount of S&P noise added to the images if
  `add_noise` is `true`.
- `noise_s_vs_p`: Sets the ratio of white and black noise in images if
  `add_noise` is `true`.

### Parameters for the `train` stage

- `validation_split`: The split ratio for the validation set, reserved from the
  training set. If this value is `0`, the test set is used for validation. 
- `epochs`: Number of epochs to train the network. 
- `batch_size`: Batch size for the `model.fit` method. 

### Parameters for the `model`

These parameters are used to set the attributes of the models. Although their
structure is fixed, you can set some important parameters that will affect the
performance, like `units` for the MLP or `conv_units` for the CNN.

- `name`: Used to select the model. For `mlp` a simple NN with a single hidden
  layer is used. For `cnn`, a Convolutional Net with a single single `Conv2D`
  and a single `Dense` layer is used. The parameters for these networks are
  defined in separate sections below.
- `optimizer`: Adam
  loss: CategoricalCrossentropy
  mlp:
    units: 16
    activation: relu
  cnn:
    dense_units: 128
    activation: relu
    conv_kernel_size: 3
    conv_units: 32
    dropout: 0.5
  metrics:
    categorical_accuracy: true
    recall: true
    precision: true
    auc-roc: true
    auc-prc: true
    fp: false
    fn: false
    tp: false
    tn: false

## Files

### Data Files

The data files used in the project are found in `data/`. All of these files are
tracked by DVC and can be retrieved using `dvc pull` from the configured remote.

- `data/raw.dvc`: Contains a reference to the [Dataset
  Registry](https://github.com/iterative/dataset-registry) to download the MNIST
  dataset to `data/raw/`.
- `data/prepared/`: Created by `src/prepare.py` and contains training and testing files in NumPy format. 
- `data/preprocessed/`: Created by `src/preprocess.py` and contains training and
  testing files in NumPy format ready to be supplied to Tensorflow.

### Source Files

The source files are `src/` directory. All files receive runtime parameters from
`params.yaml`, so none of them require any options. File dependencies are
hardcoded in the current version, but this may change in a later iteration.
Almost all capabilities of these scripts can be modified with the options in `params.yaml`

- `src/prepare.py`: Reads the raw dataset files in `data/raw/` and converts to
  NumPy format. As the MNIST dataset contains train and test sets, this script
  can remix and split them if needed. The output files are stored in
  `data/prepared/`.
  
- `src/preprocess.py`: Reads data files from `data/prepared/` and adds salt and
  pepper noise, normalize the values and shuffles. The output in
  `data/preprocessed/` is ready to supply to the Neural Network.

- `src/models.py`: Contains two models. The first one is an MLP with a single
  hidden layer.  The second is a deeper network with a convolution layer, max
  pooling, dropout, and a hidden dense layer. Various parameters of these
  networks can be set in `params.yaml`. The metrics produced as the output are
  also compiled into models in this file. The metrics can be turned on-and-off in the parameters. 

- `src/train.py`: Trains the neural network supplied by `src/models.py` with the
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
- `dvc.lock`: Parameters and dependency state is tracked with this file.

# The Pipeline

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