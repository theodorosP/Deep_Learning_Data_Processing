from keras.datasets import imdb
import numpy as np
from keras import layers, models, regularizers
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

class BestModel():
  def __init__(self):
    self.num_words = 1000
    self.train_data, self.train_target, self.test_data, self.test_target = self.get_data()
    self.x_train, self.x_test, self.y_train, self.y_test = self.one_hot_encode()
    self.train_results = self.get_best_model()
    self.test_loss_no_L2, self.test_loss_L2, self.test_accuracy_no_L2, self.test_accuracy_L2, self.ratio_no_L2, self.ratio_L2 = self.get_plotting_data()

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

  def build_model(self, num_of_neurons_1, num_of_neurons_2):
    model = models.Sequential()
    model.add( layers.Dense(num_of_neurons_1, input_shape = (self.x_train.shape[1], ), activation = "relu") ) #use input shape since we do not have any features, input_shape is the parameter matrix, maning the number of columns of the matrix
    model.add( layers.Dense(num_of_neurons_2, activation = "relu") )
    model.add( layers.Dense(1, activation = "sigmoid") ) #1 is the number of columns of the y data
    model.compile(optimizer = "rmsprop", loss = "binary_crossentropy", metrics = ["accuracy"])
    return model

  def build_model_L2(self, num_of_neurons_1, num_of_neurons_2):
    model = models.Sequential()
    model.add( layers.Dense(num_of_neurons_1, input_shape = (self.x_train.shape[1], ), activation = "relu", kernel_regularizer = regularizers.l2(0.001)  )  )
    model.add( layers.Dense(num_of_neurons_2, activation = "relu", kernel_regularizer = regularizers.l2(0.001) ) )
    model.add( layers.Dense(1, activation = "sigmoid") )
    model.compile(optimizer = "rmsprop", loss = "binary_crossentropy", metrics = ["accuracy"])
    return model

  def get_best_model( self ):
    training_results = {}
    for i in [self.build_model, self.build_model_L2] :
      for j in [ 4, 8, 16, 32 ]:
        for k in [4, 8, 16, 32]:
          model = i( j, k )
          early_stopping = EarlyStopping(monitor='val_loss', patience = 2 , restore_best_weights = True)
          history = model.fit( self.x_train, self.y_train, epochs = 30, batch_size = 512, validation_split = 0.2, verbose = False, callbacks = [early_stopping])
          #print("epochs = ", len(history.history['loss']))
          results = model.evaluate(self.x_test, self.y_test, verbose = False)
          print("test loss: ", results[0])
          print("test accuracy: ", results[1])
          print("test_loss/training_loss = ", round(results[0] / history.history["loss"][-1], 2) ) # it should usually be lower than 1.2 to avoid overfitting
          print("model = ", str( i ), " first layer: ", j, " second layer: ", k)
          if i == self.build_model:
            training_results["No_L2_" + str(j) + "_" + str(k) + "[test_loss, test_accuracy, test_loss/traing_loss]"] = [results[0], results[1], round(results[0] / history.history["loss"][-1], 2)]
            print("--------" * 10)
          elif i ==self.build_model_L2:
            training_results["L2_" + str(j) + "_" + str(k) + "[test_loss, test_accuracy, test_loss/traing_loss]"] = [results[0], results[1], round(results[0] / history.history["loss"][-1], 2)]
            print("--------" * 10)
    return training_results 

  def get_plotting_data(self):
    test_loss_no_L2 = list()
    test_loss_L2 = list()
    test_accuracy_No_L2 = list()
    test_accuracy_L2 = list()
    ratio_No_L2 = list()
    ratio_L2 = list()
    for i in [4, 8, 16, 32]:
        for j in [4, 8, 16, 32]:
            key_no_L2 = "No_L2_" + str(i) + "_" + str(j) + "[test_loss, test_accuracy, test_loss/traing_loss]"
            key_L2 = "L2_" + str(i) + "_" + str(j) + "[test_loss, test_accuracy, test_loss/traing_loss]"
            test_loss_no_L2.append(self.train_results[key_no_L2][0])
            test_loss_L2.append(self.train_results[key_L2][0])
            test_accuracy_No_L2.append(self.train_results[key_no_L2][1])
            test_accuracy_L2.append(self.train_results[key_L2][1])
            ratio_No_L2.append(self.train_results[key_no_L2][2])
            ratio_L2.append(self.train_results[key_L2][2])
    print("test_loss_no_L2 = ", test_loss_no_L2)
    print("test_loss_L2 = ", test_loss_L2)
    print("test_accuracy_No_L2 = ", test_accuracy_No_L2)
    print("test_accuracy_L2 = ", test_accuracy_L2)
    print("ratio_No_L2 = ", ratio_No_L2)
    print("ratio_L2 = ", ratio_L2)
    return test_loss_no_L2, test_loss_L2, test_accuracy_No_L2, test_accuracy_L2, ratio_No_L2, ratio_L2

  def plot_training_results(self):
    configurations = ["4-4", "4-8", "4-16", "4-32", "8-4", "8-8", "8-16", "8-32", "16-4", "16-8", "16-16", "16-32", "32-4", "32-8", "32-16", "32-32"]
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(configurations, self.test_accuracy_no_L2, marker='o', label='Accuracy Without L2 Regularization') 
    plt.plot(configurations, self.test_accuracy_L2, marker='o', label='Accuracy With L2 Regularization')
    plt.title('Test Accuracy for Different Model Configurations')
    plt.xlabel('Model Configurations')
    plt.ylabel('Test Accuracy')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(configurations, self.test_loss_no_L2, marker='o', label='Loss Without L2 Regularization') 
    plt.plot(configurations, self.test_loss_L2, marker='o', label='Loss With L2 Regularization')
    plt.title('Test Loss for Different Model Configurations')
    plt.xlabel('Model Configurations')
    plt.ylabel('Test Accuracy')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(configurations, self.ratio_no_L2, marker='o', label='test_loss/training_loss Without L2') 
    plt.plot(configurations, self.ratio_L2, marker='o', label='test_loss/training_loss With L2')
    plt.title('Test loss/Training_loss for Different Model Configurations')
    plt.xlabel('Model Configurations')
    plt.ylabel('Test Accuracy')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    

obj = BestModel()
tr_results = obj.get_best_model()
