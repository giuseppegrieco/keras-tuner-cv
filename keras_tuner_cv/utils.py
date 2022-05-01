import collections

import numpy as np
import pandas as pd


def get_metrics_std_dict(metrics_dicts):
    """Averages the metrics dictionaries to one metrics dictionary."""
    metrics = collections.defaultdict(list)
    for metrics_dict in metrics_dicts:
        for metric_name, metric_value in metrics_dict.items():
            metrics[metric_name].append(metric_value)
    std_metrics = {}
    for metric_name, metric_values in metrics.items():
        std_metrics[metric_name + "_std"] = np.std(metric_values)
    return std_metrics


def pd_inner_cv_get_result(tuner):
    num_trials = tuner.oracle.max_trials
    best_trials = tuner.oracle.get_best_trials(num_trials)
    model_results = [i for i in range(num_trials)]
    for i in range(num_trials):
        trial = best_trials[i]
        model_results[i] = {}
        model_results[i]["id"] = trial.trial_id
        for param in trial.hyperparameters.values.keys():
            model_results[i][param] = trial.hyperparameters.get(param)
        for metric in trial.metrics.metrics.keys():
            model_results[i][metric] = trial.metrics.get_history(metric)[0].value[0]
    result = pd.DataFrame(model_results)
    result.set_index("id", inplace=True)
    return result
