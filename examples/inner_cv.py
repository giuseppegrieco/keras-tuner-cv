from keras_tuner_cv import inner_cv
from keras_tuner_cv.utils import pd_inner_cv_get_result

from sklearn.model_selection import KFold
from keras_tuner.tuners import RandomSearch
from keras_tuner import Objective
from keras.layers import Normalization


# Test environment
import tensorflow as tf

tf.get_logger().setLevel("INFO")

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def build_model(hp):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(
                hp.Int("units", min_value=5, max_value=30), activation="relu"
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


tuner = inner_cv(RandomSearch)(
    build_model,
    KFold(n_splits=5, random_state=12345, shuffle=True),
    save_output=True,
    save_history=True,
    objective=Objective("val_accuracy", direction="max"),
    project_name="0",
    directory="./out/inner-cv/",
    seed=12345,
    overwrite=True,
    max_trials=2,
    preprocessor=Normalization()
)

tuner.search(
    x_train,
    y_train,
    validation_split=0.2,
    batch_size=32,
    validation_batch_size=32,
    epochs=2,
    verbose=True,
)

df = pd_inner_cv_get_result(tuner)
print(df.head())
