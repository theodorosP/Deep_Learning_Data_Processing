### do this in colab !pip install keras==2.12.0 ### 
#alternatively, use the second part of the code
import numpy as np
from keras.layers import Conv2D, Dropout, Flatten, Dense
from keras import layers, Sequential, models
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
#from scikeras.wrappers import KerasClassifier


from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

class ImageProcess():

  #define constructor
  def __init__( self, dataset_name ):
    dataset_module = getattr( __import__( 'keras.datasets', fromlist = [dataset_name]) , dataset_name )
    ( self.train_images, self.train_labels ), ( self.test_images, self.test_labels ) = dataset_module.load_data()
    X_data = [ self.train_images, self.test_images ]
    Y_data = [ self.train_labels, self.test_labels ]
    self.train_images, self.test_images = self.get_data( X_data, 1 )
    self.train_labels, self.test_labels = self.get_data( Y_data, 2 )

  #functions makes data compatible with CNN architecture, I use it only in the constructor
  def convert_data( self, data ):
    data = data.reshape( data.shape[0], data.shape[1], data.shape[2], 1)
    data = data.astype("float32") / 255
    return data

  #this function returns the train/test data in the apropriate form. Use only in the constructor.
  #Difference with convert_data() is that the get_data, uses the convert_data, to get both X or Y data, regarding the input
  #choice 1: for train_images, test_images, 2: train_labels, test_labels
  def get_data( self, data_list, choice):
    try:
      if choice == 1:
        for i in range( 0, len( data_list ) ):
          data_list[ i ] = self.convert_data( data_list[i] )
        return data_list
      elif choice == 2:
        for i in range( 0, len(data_list) ):
          data_list[ i ] = to_categorical( data_list[ i ] )
        return data_list
      else:
        raise ValueError("Invalid choice. Please choose either 1 or 2")
    except Exception as e:
      print("An error occured: ", e)

  #define the baseline model. It will be optimized with GridSearchCV
  def baseline_model( self, num_of_filters, drop_out_rate , batch_size, epochs ):
    print(f"num_of_filters: {num_of_filters}, drop_out_rate: {drop_out_rate}, batch_size: {batch_size}, epochs: {epochs}")
    model = Sequential()
    model.add( Conv2D( num_of_filters, kernel_size = ( 3, 3), activation = "relu", input_shape =  ( self.train_images.shape[1], self.train_images.shape[2] , 1 ) ) )
    model.add( Dropout( drop_out_rate ) )
    model.add( Flatten() )
    model.add( Dense( self.train_labels.shape[1], activation = "softmax") )
    model.compile( optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = ["accuracy"] )
    history = model.fit( self.train_images, self.train_labels, batch_size = batch_size, epochs = epochs, verbose = False)
    #early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose = False)
    #history = model.fit( self.train_images, self.train_labels, batch_size = batch_size, epochs = epochs, verbose = False,  callbacks=[early_stopping])
    return model

  def GridSearch( self, param_grid ):

    keras_model = KerasClassifier( build_fn = self.baseline_model, verbose = False )
    grid_search = GridSearchCV( estimator = keras_model, param_grid = param_grid, cv=2, verbose = False )
    keys = keras_model.get_params().keys()
    print("keys = ", keys)
    grid_search.fit( self.train_images, self.train_labels)
    print("Best hyperparameters:", grid_search.best_params_)


obj = ImageProcess("mnist")
obj.GridSearch( param_grid = { "num_of_filters" : [16, 32, 64], "drop_out_rate" : [0.1, 0.2, 0.3], "batch_size" : [32, 64, 128, 256, 512], "epochs" : [10, 20, 30, 40] } )



#second part
import numpy as np
from keras.layers import Conv2D, Dropout, Flatten, Dense
from keras import Sequential
from keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

class ImageProcess():

    def __init__(self, dataset_name):
        dataset_module = getattr(__import__('keras.datasets', fromlist=[dataset_name]), dataset_name)
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = dataset_module.load_data()
        self.train_images, self.test_images = self.convert_data(self.train_images), self.convert_data(self.test_images)
        self.train_labels, self.test_labels = to_categorical(self.train_labels), to_categorical(self.test_labels)

    def convert_data(self, data):
        return data.reshape(data.shape[0], data.shape[1], data.shape[2], 1).astype("float32") / 255

    def baseline_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation = "relu", input_shape = (self.train_images.shape[1], self.train_images.shape[2], 1)))
        model.add(Dropout(0.1))
        model.add(Flatten())
        model.add(Dense(10, activation="softmax"))
        model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    def GridSearch(self):
        X = self.train_images
        Y = self.train_labels
        print(X.shape)

        # Define a function that returns the baseline model
        def baseline_model_callable():
            return self.baseline_model()

        # Use the callable function when instantiating KerasClassifier
        model = KerasClassifier(model = baseline_model_callable, verbose = True)

        # Define the grid search parameters
        batch_size = [ 10, 20, 40, 60, 80, 100 ]
        epochs = [ 10, 50, 100 ]
        param_grid = dict( batch_size = batch_size, epochs=epochs )

        # Create GridSearchCV instance with KerasClassifier and parameter grid
        grid = GridSearchCV( estimator = model, param_grid = param_grid, cv = 2 )

        # Fit the grid search
        grid_result = grid.fit(X, Y)

obj = ImageProcess("mnist")
obj.GridSearch()
