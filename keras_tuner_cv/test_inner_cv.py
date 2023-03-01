import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import KFold
from sklearn.model_selection import PredefinedSplit
from keras_tuner_cv.inner_cv import inner_cv
from keras_tuner_cv.utils import pd_inner_cv_get_result



class TestInnerCvWithoutLearning(unittest.TestCase):
  """Tests for inner_cv() without learning"""

  class TestHyperModel(kt.HyperModel):

    def __init__(self,factor1,factor2):
        super().__init__()
        self.factor1 = factor1
        self.factor2 = factor2

    def build(self,hp):
      model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(1)),
        tf.keras.layers.Lambda(
          lambda x,factor1,factor2: x*factor1*factor2,
          arguments={
            'factor1' : hp.Choice('factor1',values=self.factor1),
            'factor2' : hp.Choice('factor2',values=self.factor2)
          }
        )
      ])
      model.compile(loss='mae')
      return model


  def setUp(self):
    # simple data
    self.train = np.arange(100)
    #
    self.project_name = 'test_ktcv'
    self.log_dir = './log_dir/'
    # fixed cross-validation splits
    self.cv = KFold(n_splits=2,shuffle=False)
    # hyperparameter space
    self.factor1 = [9,2]
    self.factor2 = [0.5,3.0]
    # parameters for fitting the model
    self.validation_split = 0.2
    self.shuffle = False
    self.epochs = 2
    #
    self.max_trials = 10
    # expected result
    ref = []
    for f1 in self.factor1:
      for f2 in self.factor2:
        hp = kt.HyperParameters()
        hp.values['factor1'] = f1
        hp.values['factor2'] = f2
        for i,(train_index,test_index) in enumerate(self.cv.split(self.train)):
          train_ = self.train[train_index]
          test_ = self.train[test_index]
          hypermodel = self.TestHyperModel(factor1=self.factor1,factor2=self.factor2)
          model = hypermodel.build(hp)
          model.fit(
              train_,
              train_,
              validation_split=self.validation_split,
              shuffle=self.shuffle,
              epochs=self.epochs,
              verbose=False
          )
          loss = model.evaluate(x=train_,y=train_,batch_size=len(train_),verbose=False) # training
          val_loss = model.evaluate(x=test_,y=test_,batch_size=len(test_),verbose=False) # validation
          ref.append([f1,f2,i,loss,val_loss])
    ref = pd.DataFrame(ref,columns=['f1','f2','i','loss_ref','val_loss_ref'])
    ref = ref.drop('i',axis=1).groupby(['f1','f2']).agg([pd.NamedAgg('mean',lambda x: np.mean(x)),pd.NamedAgg('std',lambda x: np.std(x,ddof=0))])
    self.ref_np = ref.reset_index().sort_values(['f1','f2']).to_numpy()

  def test_randomsearchvsgridsearch(self):
    print('\n\n----- RandomSearch vs GridSearch -----\n\n')
    # result of keras_tuner_cv.inner_cv
    tuner = inner_cv(kt.tuners.RandomSearch)(
      hypermodel=self.TestHyperModel(factor1=self.factor1,factor2=self.factor2),
      inner_cv=self.cv,
      save_output=False,
      save_history=False,
      restore_best=False,
      objective='val_loss',
      project_name=self.project_name,
      directory=self.log_dir,
      seed=42,
      overwrite=True,
      max_trials=self.max_trials
    )
    tuner.search(
      self.train,
      self.train,
      validation_split=self.validation_split,
      shuffle=self.shuffle,
      epochs=self.epochs,
      verbose=False
    )
    ktcv = pd_inner_cv_get_result(tuner,self.max_trials)
    # comparison
    ktcv_np = ktcv.drop_duplicates().sort_values(['factor1','factor2'])[['factor1','factor2','loss','loss_std','val_loss','val_loss_std']].to_numpy()
    self.assertTrue(np.all(self.ref_np == ktcv_np))

  def test_bayesianoptimizationvsgridsearch(self):
    print('\n\n----- BayesianOptimization vs GridSearch -----\n\n')
    # result of keras_tuner_cv.inner_cv
    tuner = inner_cv(kt.tuners.BayesianOptimization)(
      hypermodel=self.TestHyperModel(factor1=self.factor1,factor2=self.factor2),
      inner_cv=self.cv,
      save_output=False,
      save_history=False,
      restore_best=False,
      objective='val_loss',
      project_name=self.project_name,
      directory=self.log_dir,
      seed=42,
      overwrite=True,
      max_trials=self.max_trials,
      beta = 5
    )
    tuner.search(
      self.train,
      self.train,
      validation_split=self.validation_split,
      shuffle=self.shuffle,
      epochs=self.epochs,
      verbose=False
    )
    ktcv = pd_inner_cv_get_result(tuner,self.max_trials)
    # comparison
    ktcv_np = ktcv.drop_duplicates().sort_values(['factor1','factor2'])[['factor1','factor2','loss','loss_std','val_loss','val_loss_std']].to_numpy()
    self.assertTrue(np.all(self.ref_np == ktcv_np))

  def test_hyperbandvsgridsearch(self):
    print('\n\n----- Hyperband vs GridSearch -----\n\n')
    # result of keras_tuner_cv.inner_cv
    tuner = inner_cv(kt.tuners.Hyperband)(
      hypermodel=self.TestHyperModel(factor1=self.factor1,factor2=self.factor2),
      inner_cv=self.cv,
      save_output=False,
      save_history=False,
      restore_best=False,
      objective='val_loss',
      project_name=self.project_name,
      directory=self.log_dir,
      seed=42,
      overwrite=True
    )
    tuner.search(
      self.train,
      self.train,
      validation_split=self.validation_split,
      shuffle=self.shuffle,
      epochs=self.epochs,
      verbose=False
    )
    ktcv = pd_inner_cv_get_result(tuner,self.max_trials)
    # comparison
    ktcv_np = ktcv.drop_duplicates().sort_values(['factor1','factor2'])[['factor1','factor2','loss','loss_std','val_loss','val_loss_std']].to_numpy()
    self.assertTrue(np.all(self.ref_np == ktcv_np))



class TestInnerCvMlpMnist(unittest.TestCase):
  """Tests for inner_cv() using a multilayer perceptron and MNIST-Data, testing against same HP optimizer without cv"""
  
  class TestHyperModel(kt.HyperModel):

    def build(self,hp):
      # initialize PRNG
      tf.random.set_seed(42)
      model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(
          hp.Int('units',min_value=10,max_value=160,step=50),
          kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
          activation='relu'
        ),
        tf.keras.layers.Dense(
          10,
          kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)
        )
      ])
      model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
      )
      return model
  
  
  def setUp(self):
    # MNIST data
    # 3 identical splits
    self.n_unique = 200
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[:self.n_unique,:,:] / 255.0
    y_train = y_train[:self.n_unique]
    self.x_train3 = np.tile(x_train,(3,1,1))
    self.y_train3 = np.tile(y_train,3)
    self.x_train2 = np.tile(x_train,(2,1,1))
    self.y_train2 = np.tile(y_train,2)
    self.x_train1 = x_train
    self.y_train1 = y_train
    # 
    self.project_name = 'test_ktcv'
    self.log_dir = './log_dir/'
    # fixed cross-validation splits
    self.cv = PredefinedSplit(test_fold=np.concatenate(([1]*self.n_unique,[2]*self.n_unique,[3]*self.n_unique)))
    # parameters for fitting the model
    self.shuffle = False
    self.epochs = 3
    #
    self.max_trials = 10 
  
  def test_randomsearch(self):
    print('\n\n----- RandomSearch -----\n\n')
    # expected result
    tuner = kt.RandomSearch(
      hypermodel=self.TestHyperModel(),
      objective='val_accuracy',
      project_name=self.project_name,
      directory=self.log_dir,
      overwrite=True,
      seed=42,
      max_trials=self.max_trials
    )
    tuner.search(
      self.x_train2,
      self.y_train2,
      validation_data=[self.x_train1,self.y_train1],
      validation_batch_size=self.n_unique, # as in keras_tuner_cv
      shuffle=self.shuffle,
      epochs=self.epochs,
      verbose=False
    )
    ref_hp = tuner.get_best_hyperparameters(num_trials=self.max_trials)
    ref = []
    for r in ref_hp:
      ref.append(r.values)
    ref = pd.DataFrame(ref)
    ref_np = ref[['units']].drop_duplicates().to_numpy()
    # result of keras_tuner_cv.inner_cv
    tuner = inner_cv(kt.tuners.RandomSearch)(
      hypermodel=self.TestHyperModel(),
      inner_cv=self.cv,
      save_output=False,
      save_history=False,
      restore_best=False,
      objective='val_accuracy',
      project_name=self.project_name,
      directory=self.log_dir,
      seed=42,
      overwrite=True,
      max_trials=self.max_trials
    )
    tuner.search(
      self.x_train3,
      self.y_train3,
      validation_data=[self.x_train1,self.y_train1],
      shuffle=self.shuffle,
      epochs=self.epochs,
      verbose=False
    )
    ktcv = pd_inner_cv_get_result(tuner,self.max_trials)
    ktcv_np = ktcv[['units']].drop_duplicates().to_numpy()
    # comparison assuming same result of HPO
    self.assertTrue(np.all(ref_np == ktcv_np))

  def test_bayesianoptimization(self):
    print('\n\n----- Bayesian Optimization -----\n\n')
    # expected result
    tuner = kt.BayesianOptimization(
      hypermodel=self.TestHyperModel(),
      objective='val_accuracy',
      project_name=self.project_name,
      directory=self.log_dir,
      overwrite=True,
      seed=42,
      max_trials=self.max_trials,
      beta = 6
    )
    tuner.search(
      self.x_train2,
      self.y_train2,
      validation_data=[self.x_train1,self.y_train1],
      validation_batch_size=self.n_unique, # as in keras_tuner_cv
      shuffle=self.shuffle,
      epochs=self.epochs,
      verbose=False
    )
    ref_hp = tuner.get_best_hyperparameters(num_trials=self.max_trials)
    ref = []
    for r in ref_hp:
      ref.append(r.values)
    ref = pd.DataFrame(ref)
    ref_np = ref[['units']].drop_duplicates().to_numpy()
    # result of keras_tuner_cv.inner_cv
    tuner = inner_cv(kt.tuners.BayesianOptimization)(
      hypermodel=self.TestHyperModel(),
      inner_cv=self.cv,
      save_output=False,
      save_history=False,
      restore_best=False,
      objective='val_accuracy',
      project_name=self.project_name,
      directory=self.log_dir,
      seed=42,
      overwrite=True,
      max_trials=self.max_trials,
      beta = 6
    )
    tuner.search(
      self.x_train3,
      self.y_train3,
      validation_data=[self.x_train1,self.y_train1],
      shuffle=self.shuffle,
      epochs=self.epochs,
      verbose=False
    )
    ktcv = pd_inner_cv_get_result(tuner,self.max_trials)
    ktcv_np = ktcv[['units']].drop_duplicates().to_numpy()
    # comparison assuming same result of HPO
    self.assertTrue(np.all(ref_np == ktcv_np))

  def test_hyperband(self):
    print('\n\n----- Hyperband -----\n\n')
    # expected result
    tuner = kt.Hyperband(
      hypermodel=self.TestHyperModel(),
      objective='val_accuracy',
      project_name=self.project_name,
      directory=self.log_dir,
      overwrite=True,
      seed=42
    )
    tuner.search(
      self.x_train2,
      self.y_train2,
      validation_data=[self.x_train1,self.y_train1],
      validation_batch_size=self.n_unique, # as in keras_tuner_cv
      shuffle=self.shuffle,
      epochs=self.epochs,
      verbose=False
    )
    ref_hp = tuner.get_best_hyperparameters(num_trials=self.max_trials)
    ref = []
    for r in ref_hp:
      ref.append(r.values)
    ref = pd.DataFrame(ref)
    ref_np = ref[['units']].drop_duplicates().to_numpy()
    # result of keras_tuner_cv.inner_cv
    tuner = inner_cv(kt.tuners.Hyperband)(
      hypermodel=self.TestHyperModel(),
      inner_cv=self.cv,
      save_output=False,
      save_history=False,
      restore_best=False,
      objective='val_accuracy',
      project_name=self.project_name,
      directory=self.log_dir,
      seed=42,
      overwrite=True
    )
    tuner.search(
      self.x_train3,
      self.y_train3,
      validation_data=[self.x_train1,self.y_train1],
      shuffle=self.shuffle,
      epochs=self.epochs,
      verbose=False
    )
    ktcv = pd_inner_cv_get_result(tuner,self.max_trials)
    ktcv_np = ktcv[['units']].drop_duplicates().to_numpy()
    # comparison assuming same result of HPO
    self.assertTrue(np.all(ref_np == ktcv_np))



class TestInnerCvMultipleInputsWithoutLearning(unittest.TestCase):
  """Tests for inner_cv() using a model with multiple inputs without learning"""

  class TestHyperModel(kt.HyperModel):

    def __init__(self,factor1,factor2):
        super().__init__()
        self.factor1 = factor1
        self.factor2 = factor2

    def build(self,hp):
        in1 = tf.keras.layers.Input(shape=(1))
        in2 = tf.keras.layers.Input(shape=(1))
        con = tf.keras.layers.concatenate([in1,in2])
        out = tf.keras.layers.Lambda(
            lambda x,factor1,factor2: x*factor1*factor2,
            arguments={
                'factor1' : hp.Choice('factor1',values=self.factor1),
                'factor2' : hp.Choice('factor2',values=self.factor2)
            }
        )(in1)
        model = tf.keras.models.Model(inputs=[in1,in2],outputs=out)
        model.compile(loss='mae')
        return model


  def setUp(self):
    # simple data
    self.train1 = np.arange(100)
    self.train2 = np.arange(100)
    #
    self.project_name = 'test_ktcv'
    self.log_dir = './log_dir/'
    # fixed cross-validation splits
    self.cv = KFold(n_splits=2,shuffle=False)
    # hyperparameter space
    self.factor1 = [9,2]
    self.factor2 = [0.5,3.0]
    # parameters for fitting the model
    self.validation_split = 0.2
    self.shuffle = False
    self.epochs = 2
    #
    self.max_trials = 10
    # expected result
    ref = []
    for f1 in self.factor1:
      for f2 in self.factor2:
        hp = kt.HyperParameters()
        hp.values['factor1'] = f1
        hp.values['factor2'] = f2
        for i,(train_index,test_index) in enumerate(self.cv.split(self.train1)):
          train1_ = self.train1[train_index]
          train2_ = self.train2[train_index]
          test_ = self.train1[test_index]
          hypermodel = self.TestHyperModel(factor1=self.factor1,factor2=self.factor2)
          model = hypermodel.build(hp)
          model.fit(
              [train1_,train2_],
              train1_,
              validation_split=self.validation_split,
              shuffle=self.shuffle,
              epochs=self.epochs,
              verbose=False
          )
          loss = model.evaluate(x=[train1_,train2_],y=train1_,batch_size=len(train1_),verbose=False) # training
          val_loss = model.evaluate(x=[test_,test_],y=test_,batch_size=len(test_),verbose=False) # validation
          ref.append([f1,f2,i,loss,val_loss])
    ref = pd.DataFrame(ref,columns=['f1','f2','i','loss_ref','val_loss_ref'])
    ref = ref.drop('i',axis=1).groupby(['f1','f2']).agg([pd.NamedAgg('mean',lambda x: np.mean(x)),pd.NamedAgg('std',lambda x: np.std(x,ddof=0))])
    self.ref_np = ref.reset_index().sort_values(['f1','f2']).to_numpy()

  def test_randomsearchvsgridsearch(self):
    print('\n\n----- RandomSearch vs GridSearch -----\n\n')
    # result of keras_tuner_cv.inner_cv
    tuner = inner_cv(kt.tuners.RandomSearch)(
      hypermodel=self.TestHyperModel(factor1=self.factor1,factor2=self.factor2),
      inner_cv=self.cv,
      save_output=False,
      save_history=False,
      restore_best=False,
      objective='val_loss',
      project_name=self.project_name,
      directory=self.log_dir,
      seed=42,
      overwrite=True,
      max_trials=self.max_trials
    )
    tuner.search(
      [self.train1,self.train2],
      self.train1,
      validation_split=self.validation_split,
      shuffle=self.shuffle,
      epochs=self.epochs,
      verbose=False
    )
    ktcv = pd_inner_cv_get_result(tuner,self.max_trials)
    # comparison
    ktcv_np = ktcv.drop_duplicates().sort_values(['factor1','factor2'])[['factor1','factor2','loss','loss_std','val_loss','val_loss_std']].to_numpy()
    self.assertTrue(np.all(self.ref_np == ktcv_np))

  def test_bayesianoptimizationvsgridsearch(self):
    print('\n\n----- BayesianOptimization vs GridSearch -----\n\n')
    # result of keras_tuner_cv.inner_cv
    tuner = inner_cv(kt.tuners.BayesianOptimization)(
      hypermodel=self.TestHyperModel(factor1=self.factor1,factor2=self.factor2),
      inner_cv=self.cv,
      save_output=False,
      save_history=False,
      restore_best=False,
      objective='val_loss',
      project_name=self.project_name,
      directory=self.log_dir,
      seed=42,
      overwrite=True,
      max_trials=self.max_trials,
      beta = 5
    )
    tuner.search(
      [self.train1,self.train2],
      self.train1,
      validation_split=self.validation_split,
      shuffle=self.shuffle,
      epochs=self.epochs,
      verbose=False
    )
    ktcv = pd_inner_cv_get_result(tuner,self.max_trials)
    # comparison
    ktcv_np = ktcv.drop_duplicates().sort_values(['factor1','factor2'])[['factor1','factor2','loss','loss_std','val_loss','val_loss_std']].to_numpy()
    self.assertTrue(np.all(self.ref_np == ktcv_np))

  def test_hyperbandvsgridsearch(self):
    print('\n\n----- Hyperband vs GridSearch -----\n\n')
    # result of keras_tuner_cv.inner_cv
    tuner = inner_cv(kt.tuners.Hyperband)(
      hypermodel=self.TestHyperModel(factor1=self.factor1,factor2=self.factor2),
      inner_cv=self.cv,
      save_output=False,
      save_history=False,
      restore_best=False,
      objective='val_loss',
      project_name=self.project_name,
      directory=self.log_dir,
      seed=42,
      overwrite=True
    )
    tuner.search(
      [self.train1,self.train2],
      self.train1,
      validation_split=self.validation_split,
      shuffle=self.shuffle,
      epochs=self.epochs,
      verbose=False
    )
    ktcv = pd_inner_cv_get_result(tuner,self.max_trials)
    # comparison
    ktcv_np = ktcv.drop_duplicates().sort_values(['factor1','factor2'])[['factor1','factor2','loss','loss_std','val_loss','val_loss_std']].to_numpy()
    self.assertTrue(np.all(self.ref_np == ktcv_np))



if __name__ == '__main__':
  unittest.main()
