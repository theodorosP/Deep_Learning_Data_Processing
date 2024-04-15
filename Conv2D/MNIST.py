from keras import layers, Sequential, models
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

class DataConv2D():

    def __init__( self, dataset_name ):
        dataset_module = getattr(__import__('keras.datasets', fromlist=[dataset_name]), dataset_name)
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = dataset_module.load_data()
        data_arrays = [self.train_images, self.test_images]
        for i in range(0, len(data_arrays)):
          data_arrays[ i ] = self.convert_data( data_arrays[i] )
        self.train_images, self.test_images = data_arrays
        self.train_labels, self.test_labels = to_categorical(self.train_labels), to_categorical(self.test_labels)

    def convert_data( self, name ):
      name = name.reshape(name.shape[0], name.shape[1], name.shape[2], 1)
      name = name.astype('float32') / 255 # Standard preprocessing for Conv2D
      return name

    def evaluate_accuracy_manually(self, test_labels, prediction_labels):
      count = 0
      for i in range(0, len(test_labels)):
        if test_labels[ i ] == prediction_labels[ i ]:
          count += 1
      accuracy = count / len( test_labels )
      return accuracy

    def baseline_model(self):
      np.random.seed(42)
      model = models.Sequential()
      model.add(layers.Conv2D( 32, (3, 3), activation = "relu", input_shape = (self.train_images.shape[1], self.train_images.shape[2], 1)))
      model.add(layers.MaxPooling2D((2,2)))
      model.add(layers.Flatten())
      model.add(layers.Dense(self.train_labels.shape[1], activation = "softmax"))
      return model 

    def deep_model(self):
      np.random.seed(42)
      model = models.Sequential()
      model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(self.train_images.shape[1], self.train_images.shape[2], 1))) 
      model.add(layers.MaxPooling2D((2, 2)))
      model.add(layers.Conv2D(64, (3, 3), activation="relu"))
      model.add(layers.MaxPooling2D((2, 2)))
      model.add(layers.Conv2D(64, (3, 3), activation="relu"))
      model.add(layers.Flatten())
      model.add(layers.Dense(self.train_labels.shape[1], activation="softmax"))  # Final layer with 10 units for 10 classes
      return model

    def dense_model(self):
      np.random.seed(42)
      model = models.Sequential()
      model.add(layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (self.train_images.shape[1], self.train_images.shape[2], 1)))
      model.add(layers.MaxPooling2D(2, 2))
      model.add(layers.Flatten())
      model.add(layers.Dense(16, activation = "relu"))
      model.add(layers.Dense(32, activation = "relu"))
      model.add(layers.Dense(self.train_labels.shape[1], activation = "softmax"))
      return model

    def plot(self, variable_1, variable_2, label_1, label_2, x_label, y_label):
      plt.plot( range( 1, len( variable_1 ) + 1 ), variable_1, "-o", label = label_1 )
      plt.plot( range( 1, len( variable_2 ) + 1 ), variable_2, "-o", label = label_2 )
      plt.xlabel( x_label )
      plt.ylabel( y_label )
      plt.legend( loc = "best")
      plt.show()

    def train_model(self):
      np.random.seed(42)
      training_results = {}
      for i in [self.baseline_model, self.deep_model, self.dense_model]:
        model = i()
        model.compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = ["accuracy"])
        early_stopping = EarlyStopping(monitor='val_loss', patience = 1 , restore_best_weights = True)
        history = model.fit( self.train_images, self.train_labels, epochs = 10, batch_size = 512, validation_split = 0.2, verbose = True, callbacks = [early_stopping])
        preds = model.predict(self.test_images)      
        results = model.evaluate(self.test_images, self.test_labels)
        my_accuracy = self.evaluate_accuracy_manually( self.test_labels.argmax( axis = 1), preds.argmax(axis = 1) )
        print("keras accuracy = ", results[1] )
        print("My accuracy = ", my_accuracy)
        print("test_loss/training_loss = ", round(results[0] / history.history["loss"][-1], 2) ) 
        print("-----" * 10, str(i), "-----" * 10 )
        if i == self.baseline_model:
          training_results["baseline"] = history.history
        elif i == self.deep_model:
          training_results["deep_model"] = history.history
        elif i == self.dense_model:
          training_results["dense_model"] = history.history
      print(training_results)
      return training_results

    def plot_training_results(self, training_results_dictionary):
      v1, v2 = training_results_dictionary["baseline"]["loss"], training_results_dictionary["baseline"]["val_loss"]
      v3, v4 = training_results_dictionary["deep_model"]["loss"], training_results_dictionary["deep_model"]["val_loss"]
      v5, v6 = training_results_dictionary["dense_model"]["loss"], training_results_dictionary["dense_model"]["val_loss"]
      v7, v8 = training_results_dictionary["baseline"]["accuracy"], training_results_dictionary["baseline"]["val_accuracy"]
      v9, v10 = training_results_dictionary["deep_model"]["accuracy"], training_results_dictionary["deep_model"]["val_accuracy"]
      v11, v12 = training_results_dictionary["dense_model"]["accuracy"], training_results_dictionary["dense_model"]["val_accuracy"]
      self.plot(v1, v2, "Base_loss", "Base_val_loss", "epochs", "Base-Loss")
      self.plot(v3, v4, "Deep_loss", "Deep_val_loss", "epochs", "Deep-Loss")
      self.plot(v5, v6, "Dense_loss", "Dense_val_loss", "epochs", "Dense-Loss")
      self.plot(v7, v8, "Base_accuracy", "Base_val_accuracy", "epochs", "Base-Accuracy")
      self.plot(v9, v10, "Deep_accuracy", "Deep_val_accuracy", "epochs", "Deep-Accuracy")
      self.plot(v11, v12, "Dense_accuracy", "Dense_val_accuracy", "epochs", "Dense-Accuracy")


      

# Usage:
obj = DataConv2D("mnist")
print("train_images: ", obj.train_images.shape)
print("test_images: ", obj.test_images.shape)
print("train_labels: ", obj.train_labels.shape)
print("test_labels: ", obj.test_labels.shape)
res = obj.train_model()
obj.plot_training_results(res)
