import tensorflow as tf
import yaml

def mlp(dense_units=128, activation="relu"):
    return tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(dense_units,activation=activation),
  tf.keras.layers.Dense(10)
])

def cnn(dense_units=128, conv_kernel=(3,3), conv_units=32, dropout=0.5, activation="relu"):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(conv_units,
                               kernel_size=conv_kernel,
                               activation=activation,
                               input_shape=(28, 28)),
        tf.keras.layers.Conv2D(conv_units*2,
                               kernel_size=conv_kernel,
                               activation=activation),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(dense_units, actvation=activation),
        tf.keras.layers.Dense(10)])


def load_params():
    return yaml.safe_load(open("params.yaml"))[""]

def get_model():
    model_params = yaml.safe_load(open("params.yaml"))["model"]

    if model_params["name"].lower() == "mlp":
        p = model_params["mlp"]
        model = mlp(p["units"], p["activation"])
    elif model_params["name"].lower() == "cnn":
        p = yaml.safe_load(open("params.yaml"))["model_cnn"]
        model = cnn(dense_units=p["dense_units"],
                    conv_kernel=(p["conv_kernel_size"], p["conv_kernel_size"]),
                    conv_units=p["conv_units"],
                    dropout=p["dropout"],
                    activation=p["activation"])
    else:
        raise Excepotion(f"No Model with the name {model_params['name']} is defined")

    if model_params["optimizer"].lower() == "adam":
        optimizer = tf.keras.optimizers.Adam(0.001)
    else:
        raise Exception(f"No optimizer with the name {model_params['optimizer']} is defined")

    if model_params["loss"].lower() == "sparsecategoricalcrossentropy":
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    else:
        raise Exception(f"No loss function with the name {model_params['loss']} is defined")

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return model


