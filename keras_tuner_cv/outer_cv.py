import copy
import os

from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras_tuner.engine import tuner_utils

from keras_tuner_cv.utils import get_metrics_std_dict


class OuterCV:
    def __init__(self, outer_cv: BaseCrossValidator, tuner_class, *args, **kwargs):
        """OuterCV constructor.

        Args:
            cv (BaseCrossValidator): instance of cross validator to use.
        """
        if len(args) > 0:
            self._build_model = args[0]
        else:
            self._build_model = kwargs.get("hypermodel")
        self._outer_cv = outer_cv
        self._tuners = []
        self._output_dirs = []
        for i in range(outer_cv.get_n_splits()):
            copied_kwargs = copy.copy(kwargs)
            copied_kwargs["directory"] = os.path.join(
                kwargs["directory"], "outer_cv_" + str(i)
            )
            self._output_dirs.append(
                os.path.join(copied_kwargs["directory"], copied_kwargs["project_name"])
            )
            self._tuners.append(tuner_class(*args, **copied_kwargs))
        self._verbose = True
        self.random_state = None

    def search(self, *args, **kwargs):
        if "verbose" in kwargs:
            self._verbose = kwargs.get("verbose")

        X = args[0]
        Y = args[1]

        for split, (train_index, test_index) in enumerate(self._outer_cv.split(X, Y)):
            if self._verbose:
                tf.get_logger().info(
                    "\n" + "-" * 30 + "\n"
                    f"[Search] Outer Cross-Validation {split + 1}/{self._outer_cv.get_n_splits()}"
                    + "\n"
                    + "-" * 30
                    + "\n"
                )

            # Training split
            x_train = X[train_index]
            y_train = Y[train_index]

            copied_args, copied_kwargs = self._compute_training_args(
                x_train, y_train, *args, **kwargs
            )

            # Hyperparameter optimization
            tuner = self._tuners[split]
            tuner.search(*copied_args, **copied_kwargs)

    def evaluate(self, *args, **kwargs):
        if "verbose" in kwargs:
            self._verbose = kwargs.get("verbose")

        X = args[0]
        Y = args[1]

        results = []
        for split, (train_index, test_index) in enumerate(self._outer_cv.split(X, Y)):
            if self._verbose:
                tf.get_logger().info(
                    "\n" + "-" * 30 + "\n"
                    f"[Evaluate] Outer Cross-Validation {split + 1}/{self._outer_cv.get_n_splits()}"
                    + "\n"
                    + "-" * 30
                    + "\n"
                )

            # Training split
            x_train = X[train_index]
            y_train = Y[train_index]

            # Test split
            x_test = X[test_index]
            y_test = Y[test_index]

            # Re-fit best model found during search
            tuner = self._tuners[split]
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            model = self._build_model(best_hps)
            copied_args, copied_kwargs = self._compute_training_args(
                x_train, y_train, *args, **kwargs
            )
            model_path = os.path.join(self._output_dirs[split], "best_model")
            if not "callbacks" in copied_kwargs:
                copied_kwargs["callbacks"] = []
            copied_kwargs["callbacks"].append(
                tuner_utils.SaveBestEpoch(
                    objective=tuner.oracle.objective,
                    filepath=model_path,
                )
            )
            model.fit(*copied_args, **copied_kwargs)

            # Restore best weight according to validation score
            model = self._build_model(best_hps)
            model.load_weights(model_path).expect_partial()

            # Compute training score
            result = self._evaluate(model, copied_args[0], copied_args[1])
            # Compute validation score
            if "validation_data" in copied_kwargs:
                validation_data = copied_kwargs.get("validation_data")
                result.update(
                    self._evaluate(
                        model, validation_data[0], validation_data[1], "val_"
                    )
                )
            # Compute test score
            result.update(self._evaluate(model, x_test, y_test, "test_"))

            results.append(result)

        # Compute average score across outer folds
        result = tuner_utils.average_metrics_dicts(results)
        # Compute standard deviation across outer folds
        result.update(get_metrics_std_dict(results))

        return result

    def get_best_hparams(self):
        results = []
        for i in range(self._outer_cv.get_n_splits()):
            results.append(self._tuners[i].get_best_hyperparameters(num_trials=1)[0])
        return results

    def _evaluate(self, model, x, y, prefix=""):
        evaluation = model.evaluate(x, y, batch_size=len(x), return_dict=True)
        return {prefix + str(key): val for key, val in evaluation.items()}

    def _compute_training_args(self, x_train, y_train, *args, **kwargs):
        copied_kwargs = copy.copy(kwargs)

        if "validation_split" in kwargs:
            copied_kwargs.pop("validation_split")
            x_train, x_val, y_train, y_val = train_test_split(
                x_train,
                y_train,
                test_size=kwargs.get("validation_split"),
                random_state=self.random_state,
            )
            copied_kwargs["validation_data"] = (x_val, y_val)
            if (
                "validation_batch_size" in kwargs
                and kwargs.get("validation_batch_size") == "full-batch"
            ):
                copied_kwargs["validation_batch_size"] = len(x_val)

        # Compute full-batch size for training data
        if "batch_size" in kwargs and kwargs.get("batch_size") == "full-batch":
            copied_kwargs["batch_size"] = len(x_train)

        copied_args = []
        for arg in args:
            copied_args.append(arg)
        copied_args[0] = x_train
        copied_args[1] = y_train
        copied_args = tuple(arg for arg in copied_args)

        return copied_args, copied_kwargs
