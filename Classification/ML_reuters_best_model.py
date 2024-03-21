from keras.datasets import reuters
import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

class Dataset:
  #define empty constructor
  def __init__(self):
    self.word_limit = 10000
    self.train_data, self.train_labels, self.test_data, self.test_labels = self.get_data()
    self.x_train =  self.to_one_hot(self.train_data)
    self.x_test = self.to_one_hot(self.test_data)
    self.y_train = to_categorical(self.train_labels)
    self.y_test = to_categorical(self.test_labels)

  def get_data(self):
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = self.word_limit) #keep up to 10000 most frequently used words
    return train_data, train_labels, test_data, test_labels

  def to_one_hot(self, array_to_diagonalize):
    result = np.zeros( (len(array_to_diagonalize), self.word_limit ) )
    for i, j in enumerate(array_to_diagonalize):
      result[i, j] = 1
    return result

  def plot(self, variable_1, variable_2, label_1, label_2, x_label, y_label):
      plt.plot( range(1, len(variable_1) + 1), variable_1, "-o", label = label_1)
      plt.plot( range(1, len(variable_2) + 1), variable_2, "-o", label = label_2 )
      plt.xlabel(x_label)
      plt.ylabel(y_label)
      plt.legend(loc = "best")
      plt.show()

  def my_model(self, opt, pat, epoch, bat_size):
    model = models.Sequential()
    model.add( layers.Dense( 64, input_shape = (self.x_train.shape[1], ), activation = "relu" ) )
    model.add( layers.Dense( 64, activation = "relu") )
    model.add( layers.Dense (self.y_train.shape[1], activation = "softmax") )
    model.compile(optimizer = opt, loss = "categorical_crossentropy", metrics = ["accuracy"])
    early_stopping = EarlyStopping(monitor='val_loss', patience = pat , restore_best_weights = True)
    history = model.fit(self.x_train, self.y_train, epochs = epoch, batch_size = bat_size, validation_split = 0.2, callbacks = [early_stopping], verbose = 0)
    results = model.evaluate(self.x_test, self.y_test, verbose = 0)
    loss, accuracy = results[0], results[1]
    print("Loss: ", results[0], "Accuracy: ", results[1])
    ratio = round(results[0] / history.history["loss"][-1], 2)
    print("test_loss/training_loss = ",  ratio)
    #self.plot(variable_1 = history.history["loss"], variable_2 = history.history["val_loss"], label_1 = "training loss", label_2 = "validation loss", x_label = "Epochs", y_label = "Loss" )
    #self.plot(variable_1 = history.history["accuracy"], variable_2 = history.history["val_accuracy"], label_1 = "training accuracy", label_2 = "validation accuracy", x_label = "Epochs", y_label = "Accuracy" )
    return round(accuracy, 3), round(loss, 3), round(results[0] / history.history["loss"][-1], 2)

  def get_optimal_model(self):
    optimal_results = {}
    results = list()
    for i in ["SGD", "AdaDelta", "AdaGrad", "RMSProp", "Adam"]:
      for j in range(1, 7):
        for k in [1, 5, 10, 15, 20]: #range(1, 20):
          for l in [32, 64, 128, 256, 512]:
            print("Evaluating optimizer: ", i, " patience: ", j, " epochs: ", k, " batch_size: ", l)
            accuracy, loss, ratio = self.my_model(i, j, k, l) 
            optimal_results = {"accuracy" : accuracy,  "loss": loss, "ratio" : ratio, "optimizer" : i, "patience" : j, "epochs" : k, "batch_size" : l}
            results.append(optimal_results)
            print("----" * 20)
    for i in results:
      print(i)       


obj = Dataset()
obj.get_optimal_model()
