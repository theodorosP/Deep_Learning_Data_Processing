from keras.datasets import boston_housing
from keras import models
from keras import layers
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

class BostonHousing():
    #define constructor
    def __init__(self):
        self.train_data, self.train_target, self.test_data, self.test_target = self.get_data()
        self.train_data_normalized, self.test_data_normalized = self.normalize_data()

    def get_data(self):
      (train_data, train_target), (test_data, test_target) = boston_housing.load_data()
      return train_data, train_target, test_data, test_target

    def normalize_data(self):
        #get mean and std of train data
        mean = self.train_data.mean(axis = 0)
        self.train_data -= mean
        std = self.train_data.std(axis = 0)
        self.train_data /= std
        #use mean and std of trained data in test data
        self.test_data -= mean
        self.test_data /= std
        return self.train_data, self.test_data

    def plot(self, variable_1, variable_2, label_1, label_2, x_label, y_label):
      plt.plot( range( 1, len(variable_1) + 1 ) , variable_1, "-o", label = label_1 )
      plt.plot( range( 1, len(variable_2) + 1 ), variable_2, "-o", label = label_2)
      plt.legend(loc = "best")
      plt.xlabel( x_label )
      plt.ylabel( y_label )
      plt.show()

    def build_model(self):
      model = models.Sequential()
      model.add( layers.Dense(64, input_shape = (self.train_data_normalized.shape[1] ,), activation = "relu" ) )
      model.add(layers.Dense( 64, activation = "relu"))
      model.add( layers.Dense(1) ) #linear activation function
      model.compile( optimizer = "rmsprop", loss = "mse", metrics = ["mae"])
      return model

    def train_model_kfold(self):
      #need to import: from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
      model = KerasRegressor(build_fn=self.build_model , epochs=100, batch_size=10, verbose=0)
      early_stopping = EarlyStopping( monitor = "val_loss", patience = 3, mode = "min", restore_best_weights = True)
      kfold = KFold(n_splits=10, shuffle=True)
      scores = cross_val_score(model, self.train_data_normalized, self.train_target, cv = kfold, scoring = "neg_mean_squared_error")
      print(scores)

    def train_model_kfold_manually(self):
      best_model = None
      best_val_loss = float("inf")
      kf = KFold(n_splits = 5, shuffle = True)
      for train_index, val_index in kf.split(self.train_data_normalized):
        x_train, x_val = self.train_data_normalized[ train_index ], self.train_data_normalized[ val_index ]
        y_train, y_val = self.train_data_normalized [ train_index ], self.train_data_normalized[ val_index ]
        model = self.build_model()
        early_stopping = EarlyStopping(monitor = "val_loss", patience = 3, mode = "min", restore_best_weights = True)
        history = model.fit(x_train, y_train, epochs = 1000, batch_size = 100, validation_data = (x_val, y_val), callbacks = [early_stopping], verbose = 0 )
        train_results = model.evaluate(x_train, y_train, verbose = 0)
        val_loss = history.history["val_loss"][-1]
        print(val_loss)
        if val_loss < best_val_loss:
          best_val_loss = val_loss
          best_model = model
      print("best_val_loss: ", best_val_loss )
      return best_model

    def train_model(self):
      model = self.build_model()
      early_stopping = EarlyStopping(monitor = "val_loss", patience = 3, mode = "min", restore_best_weights = True)
      history = model.fit(self.train_data_normalized, self.train_target, epochs = 1000, batch_size = 100, validation_split = 0.2, callbacks = [early_stopping], verbose = False)
      results = model.evaluate(self.test_data_normalized, self.test_target, verbose = False)
      #in results the first variable is the test_loss (mean squared error). The second gives the test_mean_average_error (mean average error)
      self.plot(history.history["loss"], history.history["val_loss"], "training loss", "validation loss", "Loss", "Epochs")
      self.plot(history.history["mae"], history.history["val_mae"], "training loss", "validation loss", "MAE", "Epochs")

    #visualize
    def print_data(self):
      print( self.train_model() )

obj = BostonHousing()
obj.train_model_kfold_manually()
