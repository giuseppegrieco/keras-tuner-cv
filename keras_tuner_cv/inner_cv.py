import copy
import warnings
import os
import json

import tensorflow as tf

import numpy as np

from sklearn.model_selection import BaseCrossValidator

from keras_tuner.engine.tuner import Tuner, maybe_distribute
from keras_tuner.engine import tuner_utils
from keras_tuner.engine import trial as trial_module

from keras_tuner_cv.utils import get_metrics_std_dict


def inner_cv(
    superclass: Tuner,
):
    class InnerCV(superclass):
        """
        Hyparameters search evaluated using cross-validation over a
        parameter space.
        """

        def __init__(
            self,
            hypermodel,
            inner_cv: BaseCrossValidator,
            *args,
            save_history=False,
            save_output=False,
            restore_best=True,
            **kwargs,
        ):
            """TunerCV constructor.

            Args:
                cv (BaseCrossValidator): instance of cross validator to use.
            """
            super(InnerCV, self).__init__(hypermodel, *args, **kwargs)
            self._inner_cv = inner_cv
            self._save_history = save_history
            self._save_output = save_output
            self._restore_best = restore_best
            self._verbose = True

        def search(self, *fit_args, **fit_kwargs):
            if "verbose" in fit_kwargs:
                self._verbose = fit_kwargs.get("verbose")
            self.on_search_begin()
            while True:
                trial = self.oracle.create_trial(self.tuner_id)
                if trial.status == trial_module.TrialStatus.STOPPED:
                    # Oracle triggered exit.
                    tf.get_logger().info("Oracle triggered exit")
                    break
                if trial.status == trial_module.TrialStatus.IDLE:
                    # Oracle is calculating, resend request.
                    continue

                self.on_trial_begin(trial)
                results = self.run_trial(trial, *fit_args, **fit_kwargs)
                # `results` is None indicates user updated oracle in `run_trial()`.
                if results is None:
                    warnings.warn(
                        "`Tuner.run_trial()` returned None. It should return one of "
                        "float, dict, keras.callbacks.History, or a list of one "
                        "of these types. The use case of calling "
                        "`Tuner.oracle.update_trial()` in `Tuner.run_trial()` is "
                        "deprecated, and will be removed in the future.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                else:
                    metrics = tuner_utils.convert_to_metrics_dict(
                        results, self.oracle.objective, "Tuner.run_trial()"
                    )
                    metrics.update(get_metrics_std_dict(results))
                    self.oracle.update_trial(
                        trial.trial_id,
                        metrics,
                    )
                self.on_trial_end(trial)
            self.on_search_end()

        def run_trial(self, trial, *args, **kwargs):
            original_callbacks = kwargs.pop("callbacks", [])

            X = args[0]
            Y = args[1]

            # Run the training process multiple times.
            histories = []
            for execution in range(self.executions_per_trial):
                # Run the training over different splits.
                for split, (train_index, val_index) in enumerate(
                    self._inner_cv.split(X, Y)
                ):
                    if self._verbose:
                        tf.get_logger().info(
                            "\n" + "-" * 30 + "\n"
                            f"Inner Cross-Validation {split + 1}/{self._inner_cv.get_n_splits()}"
                            + "\n"
                            + "-" * 30
                            + "\n"
                        )
                    # Create a copy of args and kwargs to fill with fold-specific data
                    copied_args = []
                    copied_kwargs = copy.copy(kwargs)

                    # Get training set
                    x_train = X[train_index]
                    y_train = Y[train_index]
                    # Set the training set
                    for arg in args:
                        copied_args.append(arg)
                    copied_args[0] = x_train
                    copied_args[1] = y_train
                    copied_args = tuple(arg for arg in copied_args)
                    # If requested it sets full batch for training
                    if "batch_size" in kwargs and (
                        kwargs["batch_size"] == "full-batch"
                        or kwargs["batch_size"] > len(x_train)
                    ):
                        copied_kwargs["batch_size"] = len(x_train)

                    # Get the validation set
                    x_val = X[val_index]
                    y_val = Y[val_index]
                    # Set the validation set
                    copied_kwargs["validation_data"] = [x_val, y_val]
                    # If requested it sets full batch for validation
                    if "validation_batch_size" in kwargs and (
                        kwargs["validation_batch_size"] == "full-batch"
                        or kwargs["validation_batch_size"] > len(x_val)
                    ):
                        copied_kwargs["validation_batch_size"] = len(x_val)

                    # -------------------------------------------------------
                    # Callbacks
                    # -------------------------------------------------------

                    # Configure tensorboard
                    callbacks = self._deepcopy_callbacks(original_callbacks)
                    self._configure_tensorboard_dir(
                        callbacks, trial, str(execution) + "_" + str(split)
                    )
                    callbacks.append(tuner_utils.TunerCallback(self, trial))
                    # Save all the checkpoint.
                    # The file name will be checkpoint_{execution}_{split}
                    callbacks.append(
                        tuner_utils.SaveBestEpoch(
                            objective=self.oracle.objective,
                            filepath=self._get_checkpoint_fname(trial.trial_id)
                            + "_"
                            + str(execution)
                            + "_"
                            + str(split),
                        )
                    )
                    copied_kwargs["callbacks"] = callbacks

                    # Build and train the model
                    history, model = self._build_and_fit_model(
                        trial, *copied_args, **copied_kwargs
                    )

                    if self._restore_best:
                        # Load the best epoch according to objective function
                        model = self._try_build(trial.hyperparameters)
                        model.load_weights(
                            self._get_checkpoint_fname(trial.trial_id)
                            + "_"
                            + str(execution)
                            + "_"
                            + str(split)
                        ).expect_partial()

                    trial_path = self.get_trial_dir(trial.trial_id)
                    # Save the history if requested
                    if self._save_history:
                        self.__save_history(
                            history,
                            self.__get_filename_path(
                                trial_path, "history", ".json", execution, split
                            ),
                        )
                    # Save the output in numpy format if requested
                    if self._save_output:
                        self.__save_output(
                            model,
                            x_train,
                            self.__get_filename_path(
                                trial_path, "training", ".npy", execution, split
                            ),
                        )
                        self.__save_output(
                            model,
                            x_val,
                            self.__get_filename_path(
                                trial_path, "validation", ".npy", execution, split
                            ),
                        )

                    # Evaluate train performance on best epoch
                    obj_value = model.evaluate(
                        x_train,
                        y_train,
                        batch_size=len(x_train),
                        return_dict=True,
                        verbose=self._verbose,
                    )

                    # Evaluate validation performance on best epoch
                    val_res = model.evaluate(
                        x_val,
                        y_val,
                        batch_size=len(x_val),
                        return_dict=True,
                        verbose=self._display.verbose,
                    )
                    obj_value.update(
                        {"val_" + str(key): val for key, val in val_res.items()}
                    )

                    # Append training and validation scores to the histories
                    histories.append(obj_value)
            # It will returns an array of dictionary, note by default keras-tuner
            # will compute an average. This average is therefore the average of the
            # scores across the folds.
            return histories

        def __get_filename_path(self, trial_path, name, ext, execution, split):
            return os.path.join(
                trial_path,
                name + "_" + str(execution) + "_" + str(split) + ext,
            )

        def get_history(self, trial):
            histories = []
            trial_path = self.get_trial_dir(trial.trial_id)
            for split in range(self._inner_cv.get_n_splits()):
                executions = []
                for execution in range(self.executions_per_trial):
                    with open(
                        self.__get_filename_path(
                            trial_path, "history", ".json", execution, split
                        )
                    ) as fp:
                        executions.append(json.load(fp))
                histories.append(executions if len(executions) > 1 else executions[0])
            return histories

        def get_output(self, trial):
            outputs = []
            trial_path = self.get_trial_dir(trial.trial_id)
            for split in range(self._inner_cv.get_n_splits()):
                executions = []
                for execution in range(self.executions_per_trial):
                    training = np.load(
                        self.__get_filename_path(
                            trial_path, "training", ".npy", execution, split
                        )
                    )
                    validation = np.load(
                        self.__get_filename_path(
                            trial_path, "validation", ".npy", execution, split
                        ),
                    )
                    executions.append((training, validation))
                outputs.append(executions if len(executions) > 1 else executions[0])
            return outputs

        def _build_and_fit_model(self, trial, *args, **kwargs):
            hp = trial.hyperparameters
            model = self._try_build(hp)
            return self.hypermodel.fit(hp, model, *args, **kwargs), model

        def __save_output(self, model, x, filename):
            y = model.predict(
                x,
                batch_size=len(x),
                verbose=self._display.verbose,
            )
            with open(
                filename,
                "wb",
            ) as fp:
                np.save(fp, y)

        def __save_history(self, history, filename):
            with open(
                filename,
                "w",
            ) as fp:
                json.dump(history.history, fp)

        def load_model(self, trial):
            """
            Returns all models associated with a specific trial. The output is an array where
            the number is determined by the number of splits of the cross validation. Each
            element of the array can be a single model if self.executions_per_trial is equal
            to 1, an array if it is greater.
            """
            models = []
            for split in range(self._inner_cv.get_n_splits()):
                executions = []
                for execution in range(self.executions_per_trial):
                    model = self._try_build(trial.hyperparameters)
                    # Reload best checkpoint.
                    # Only load weights to avoid loading `custom_objects`.
                    with maybe_distribute(self.distribution_strategy):
                        model.load_weights(
                            self._get_checkpoint_fname(trial.trial_id)
                            + "_"
                            + str(execution)
                            + "_"
                            + str(split)
                        )
                    executions.append(model)
                models.append(executions if len(executions) > 1 else executions[0])
            return model

    return InnerCV
