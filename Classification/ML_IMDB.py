from keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers

class Dataset:
  #define constructor
  def __init__(self):
    self.train_data, self.train_labels, self.test_data, self.test_labels = self.get_data()
    self.x_train, self.x_test, self.y_train, self.y_test = self.one_hot_encode()

  def get_data(self):
    from keras.datasets import imdb
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)
    return train_data, train_labels, test_data, test_labels

  def diagonalize(self, matrix):
    num_words = 10000 # this is the cut off we have used
    result = np.zeros( (len(matrix), num_words) )
    for i, j in enumerate(matrix):
      result[i, j] = 1
    return result

  def one_hot_encode(self):
    x_train = self.diagonalize( self.train_data )
    x_test = self.diagonalize( self.test_data )
    y_train = np.asarray( self.train_labels )
    y_test = np.asarray( self.test_labels )
    return x_train, x_test, y_train, y_test

  def plot(self, variable_1, variable_2, label_1, label_2, x_label, y_label):
      plt.plot(variable_1, label = label_1)
      plt.plot(variable_2, label = label_2)
      plt.xlabel(x_label)
      plt.ylabel(y_label)
      plt.legend(loc = "best")
      plt.show()

  def my_model(self):
    model = models.Sequential()
    model.add( layers.Dense(16, input_shape = (self.x_train.shape[1], ), activation = "relu") ) #use input shape since we do not have any features, input_shape is the parameter matrix, maning the number of columns of the matrix
    model.add( layers.Dense(16, activation = "relu") )
    model.add( layers.Dense(1, activation = "sigmoid") ) #1 is the number of columns of the y data
    model.compile(optimizer = "rmsprop", loss = "binary_crossentropy", metrics = ["accuracy"])
    history = model.fit( self.x_train, self.y_train, epochs = 4, batch_size = 512, validation_split = 0.2)
    print(history.history.keys())
    results = model.evaluate(self.x_test, self.y_test)
    print("test loss: ", results[0])
    print("test accuracy: ", results[1])
    print("test_loss/training_loss = ", round(results[0] / history.history["loss"][-1], 2) ) # it should usually be lower than 1.2 to avoid overfitting
    #predictions = model.predict(selfx_test) #or make predictions on another dataset
    #plot the losses
    self.plot(variable_1 = history.history["loss"], variable_2 = history.history["val_loss"], label_1 = "training loss", label_2 = "validation loss", x_label = "Epochs", y_label = "Loss" )
    self.plot(variable_1 = history.history["accuracy"], variable_2 = history.history["val_accuracy"], label_1 = "training accuracy", label_2 = "validation accuracy", x_label = "Epochs", y_label = "Accuracy" )

obj = Dataset()
train_data, train_label, test_data, test_label = obj.get_data()
x_train, x_test, y_train, y_test = obj.one_hot_encode()
obj.my_model()
