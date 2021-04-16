import tensorflow as tf
import yaml

def mlp(dense_units=128, activation="relu"):
    return tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(dense_units, activation=activation),
      tf.keras.layers.Dense(10, activation="softmax")
])

def cnn(dense_units=128, conv_kernel=(3,3), conv_units=32, dropout=0.5, activation="relu"):
    return tf.keras.models.Sequential([
        tf.keras.layers.Reshape(input_shape=(28, 28),
                                target_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(conv_units,
                               kernel_size=conv_kernel,
                               activation=activation),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(dense_units, activation=activation),
        tf.keras.layers.Dense(10, activation="softmax")])


def load_params():
    return yaml.safe_load(open("params.yaml"))[""]

def get_model():
    model_params = yaml.safe_load(open("params.yaml"))["model"]

    if model_params["name"].lower() == "mlp":
        p = model_params["mlp"]
        model = mlp(p["units"], p["activation"])
    elif model_params["name"].lower() == "cnn":
        p = model_params["cnn"]
        model = cnn(dense_units=p["dense_units"],
                    conv_kernel=(p["conv_kernel_size"], p["conv_kernel_size"]),
                    conv_units=p["conv_units"],
                    dropout=p["dropout"],
                    activation=p["activation"])
    else:
        raise Exception(f"No Model with the name {model_params['name']} is defined")

    if model_params["optimizer"].lower() == "adam":
        optimizer = tf.keras.optimizers.Adam(0.001)
    else:
        raise Exception(f"No optimizer with the name {model_params['optimizer']} is defined")

    if model_params["loss"].lower() == "categoricalcrossentropy":
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    else:
        raise Exception(f"No loss function with the name {model_params['loss']} is defined")

    metrics_p = model_params["metrics"]
    metrics = []

    if metrics_p["categorical_accuracy"]:
        metrics.append(tf.keras.metrics.CategoricalAccuracy())
    if metrics_p["precision"]:
        metrics.append(tf.keras.metrics.Precision())
    if metrics_p["recall"]:
        metrics.append(tf.keras.metrics.Recall())
    if metrics_p["roc"]:
        metrics.append(tf.keras.metrics.AUC(curve="ROC", name="ROC", multi_label=True))


    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

    return model


