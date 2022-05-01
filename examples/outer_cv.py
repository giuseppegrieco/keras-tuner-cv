from keras_tuner_cv.outer_cv import OuterCV

from sklearn.model_selection import KFold
from keras_tuner.tuners import RandomSearch
from keras_tuner import Objective

# Test environment
import tensorflow as tf

tf.get_logger().setLevel("INFO")

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


def build_model(hp):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(
                hp.Int("units", min_value=128, max_value=256), activation="relu"
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


tuner = OuterCV(
    KFold(n_splits=5, random_state=12345, shuffle=True),
    RandomSearch,
    build_model,
    objective=Objective("val_accuracy", direction="max"),
    project_name="0",
    directory="./out/outer-cv/",
    seed=12345,
    overwrite=False,
    max_trials=2,
)

tuner.search(
    x_train,
    y_train,
    validation_split=0.2,
    batch_size="full-batch",
    validation_batch_size="full-batch",
    epochs=2,
    verbose=True,
)

eval = tuner.evaluate(
    x_train,
    y_train,
    validation_split=0.2,
    batch_size="full-batch",
    validation_batch_size="full-batch",
    epochs=2,
    verbose=True,
)
print(eval)
