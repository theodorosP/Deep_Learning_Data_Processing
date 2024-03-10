from keras.datasets import imdb

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
    train_data = self.get_data()[0]
    test_data = self.get_data()[2]
    return self.diagonalize(train_data), self.diagonalize(test_data)

obj = Dataset()
train_data, train_label, test_data, test_label = obj.get_data()
diag_train, diag_test = obj.one_hot_encode()
