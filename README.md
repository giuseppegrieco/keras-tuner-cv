# Keras Tuner Cross Validation
Extension for keras tuner that adds a set of classes to implement cross validation methodologies.

## Install
```
$ pip install keras_tuner_cv
```

## Implemented methodologies
Here is the list of implemented methodologies and how to use them!
### Outer Cross Validation

```python
from keras_tuner_cv.outer_cv import OuterCV

from keras_tuner.tuners import RandomSearch

from sklearn.model_selection import KFold

cv = KFold(n_splits=5, random_state=12345, shuffle=True),

outer_cv = OuterCV(
    # You can use any class extendind:
    # sklearn.model_selection.cros.BaseCrossValidator
    cv,
    # You can use any class extending:
    # keras_tuner.engine.tuner.Tuner, e.g. RandomSearch
    RandomSearch,
    # Tuner parameters both positional and named ones
    ...
)
```
### Inner Cross Validation
```python
from keras_tuner_cv.outer_cv import OuterCV

from keras_tuner.tuners import RandomSearch

from sklearn.model_selection import KFold

cv = KFold(n_splits=5, random_state=12345, shuffle=True),
    
# You can use any class extending:
# keras_tuner.engine.tuner.Tuner, e.g. RandomSearch
outer_cv = inner_cv(RandomSearch)(
    hypermodel,
    # You can use any class extendind:
    # sklearn.model_selection.cros.BaseCrossValidator
    cv,
    # Tuner positional parameters except hypermodel
    ...,
    # Saves the history of all metrics observed across the epochs 
    # in json format.    
    save_history=False,
    # Saves the model output for both the training and validation 
    # datasets in numpy format.
    save_output=False,
    # Indicates when or not to reload the best weights w.r.t. to 
    # the objective indicated for the calculation of output and
    # scores.
    restore_best=True,
    # Tuner named parameters except hypermodel
    ...
)
```

## License
Keras Tuner CV is released under the [GPL v3](LICENSE).