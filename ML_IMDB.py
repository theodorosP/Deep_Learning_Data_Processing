from keras.datasets import imdb
import numpy as np

class Dataset:
  #define constructor
  def __init__(self):
    pass

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
    x_train = self.diagonalize( self.get_data()[0] )
    x_test = self.diagonalize( self.get_data()[2] )
    y_train = np.asarray( self.get_data()[1] )
    y_test = np.asarray( self.get_data()[3] )
    return x_train, x_test, y_train, y_test 
