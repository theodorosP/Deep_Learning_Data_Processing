from keras.datasets import imdb
import numpy as np
from keras import layers, models
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

class BestModel():
  def __init__(self):
    self.num_words = 1000
    self.train_data, self.train_target, self.test_data, self.test_target = self.get_data()
    self.x_train, self.x_test, self.y_train, self.y_test = self.one_hot_encode()

  def get_data(self):
    (train_data, train_target), (test_data, test_target) = imdb.load_data( num_words = self.num_words)
    return train_data, train_target, test_data, test_target

  def diagonalize(self, matrix):
    results = np.zeros( (matrix.shape[0], self.num_words ) )
    for i, j in enumerate(matrix):
      results[i, j] = 1
    return results

  def one_hot_encode( self ):
    x_train = self.diagonalize( self.train_data )
    x_test = self.diagonalize( self.test_data )
    y_train = self.train_target
    y_test = self.test_target
    return x_train, x_test, y_train, y_test

  def plot(self, variable_1, variable_2, label_1, label_2, x_label, y_label):
      plt.plot( range(1, len(variable_1) + 1 ), variable_1, "-o", label = label_1)
      plt.plot( range(1, len(variable_2) + 1 ), variable_2, "-o", label = label_2)
      plt.xlabel(x_label)
      plt.ylabel(y_label)
      plt.legend(loc = "best")
      plt.show()

  def build_model(self, num_of_neurons):
    model = models.Sequential()
    model.add( layers.Dense(num_of_neurons, input_shape = (self.x_train.shape[1], ), activation = "relu") ) #use input shape since we do not have any features, input_shape is the parameter matrix, maning the number of columns of the matrix
    model.add( layers.Dense(num_of_neurons, activation = "relu") )
    model.add( layers.Dense(1, activation = "sigmoid") ) #1 is the number of columns of the y data
    model.compile(optimizer = "rmsprop", loss = "binary_crossentropy", metrics = ["accuracy"])
    return model

  def train_model( self ):
    loss_accuracy = {}
    for i in [ 4, 8, 16, 32]:
      model = self.build_model( i )
      early_stopping = EarlyStopping(monitor='val_loss', patience = 2 , restore_best_weights = True)
      history = model.fit( self.x_train, self.y_train, epochs = 30, batch_size = 512, validation_split = 0.2, verbose = False, callbacks = [early_stopping])
      print(history.history.keys())
      #print("epochs = ", len(history.history['loss']))
      results = model.evaluate(self.x_test, self.y_test, verbose = False)
      print("test loss: ", results[0])
      print("test accuracy: ", results[1])
      print("number of nodes: ", i,  " test_loss/training_loss = ", round(results[0] / history.history["loss"][-1], 2) ) # it should usually be lower than 1.2 to avoid overfitting
      print("----" * 10)
      loss_accuracy[ str(i) + "_neurons" ] = history.history
      print( loss_accuracy[ str(i) + "_neurons"]["loss"] )
    plt.plot( range(1, len( loss_accuracy["4_neurons"]["val_loss"]) + 1) , loss_accuracy["4_neurons"]["val_loss"], "-o", label = "4_neurons_val_loss")
    plt.plot( range(1, len( loss_accuracy["8_neurons"]["val_loss"]) + 1) , loss_accuracy["8_neurons"]["val_loss"], "-o", label = "8_neurons_val_loss")
    plt.plot( range(1, len( loss_accuracy["16_neurons"]["val_loss"]) + 1) , loss_accuracy["16_neurons"]["val_loss"], "-o", label = "16_neurons_val_loss")
    plt.plot( range(1, len( loss_accuracy["32_neurons"]["val_loss"]) + 1) , loss_accuracy["32_neurons"]["val_loss"], "-o", label = "32_neurons_val_loss")
    plt.legend( loc = "best")
    plt.xlabel("Epochs")
    plt.ylabel("Val Loss")
    plt.show()


obj = BestModel()
obj.train_model()
