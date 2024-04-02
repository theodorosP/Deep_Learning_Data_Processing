import os
import shutil
import random as random
from keras import optimizers
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras import layers, Sequential, models
from keras.preprocessing.image import ImageDataGenerator


class CatVsDogs():
    #define constructor
    def __init__(self):
        self.train_dir = "/content/drive/MyDrive/dogs-vs-cats/train"
        self.test_dir = "/content/drive/MyDrive/dogs-vs-cats/test"
        self.train_generator, self.test_generator = self.generators()
        self.data_dir = "/content/drive/MyDrive/dogs-vs-cats/train/train/"
        self.cat_dog_dir = "/content/drive/MyDrive/dogs-vs-cats/"
        self.val_ratio = 0.25

    #name is either cat or dog
    #start_index and end_index are the indexes of picures we want to visualize
    def view_as_subplot( self, name, start_index, end_index ):
        try:
          if abs( start_index - end_index ) <= 9:
            for i in range( start_index, end_index ):
              plt.subplot( 330  + i - start_index + 1  )
              filename = self.data_dir + name.lower() + "." + str( i ) + ".jpg"
              image = mpimg.imread( filename )
              plt.imshow( image )
            plt.show()
          else:
            raise ValueError( "You are allowed to visualize up to 9 images" )
        except Exception as e:
          print("An error occured", e)

    #name should be either cat or dog
    def visualize_pics( self, name, start_index, end_index ):
        try:
            if name.lower() == "cat":
                self.view_as_subplot( name.lower(), start_index, end_index )
            elif name.lower() == "dog":
                self.view_as_subplot( name.lower(), start_index, end_index )
            else:
                raise ValueError( "The name should be either dog or cat" )
        except Exception as e:
            print("An error occurred:", e)

    #make the folders for training and testing
    def make_folders( self ):
      train_test = [ "train/", "test/" ]
      dogs_cats = [ "dogs/", "cats/" ]
      for i in train_test:
        for j in dogs_cats:
          new_dir = self.cat_dog_dir + i + j
          if os.path.exists( new_dir ):
            print( "Folder exists", new_dir )
          else:
            os.makedirs( new_dir )

    #copy the images to train and test directory, 25% will be kept for testing
    def copy_images( self ):
      random.seed( 1 )
      train_dir = "/content/drive/MyDrive/dogs-vs-cats/train"
      test_dir = "/content/drive/MyDrive/dogs-vs-cats/test"
      for i in os.listdir( self.data_dir ):
        destination_dir = train_dir
        if random.random() < self.val_ratio:
          destination_dir = test_dir
        if i.startswith( "cat" ):
          file_to_copy = self.data_dir + i
          destination_dir = destination_dir + "/cats"
        elif i.startswith( "dog" ):
          file_to_copy = self.data_dir + i
          destination_dir = destination_dir + "/dogs"
        os.system( "cp " + file_to_copy + " " + destination_dir)
        #shutil.copyfile( file_to_copy, destination_dir)
        print("File ", i, " copied at: ", destination_dir)
        print("---" * 10)

    def check_data( self ):
      for i in ["train", "test"]:
        for j in ["cats", "dogs"]:
          print("/content/drive/MyDrive/dogs-vs-cats/" + i + "/" + j)
          print("----" * 10)
          for k in os.listdir("/content/drive/MyDrive/dogs-vs-cats/" + i + "/" + j):
            print( k )

    def generators( self ):
      train_datagen = ImageDataGenerator( rescale = 1./255)
      test_datagen = ImageDataGenerator( rescale = 1./255)
      train_generator = train_datagen.flow_from_directory( self.train_dir, target_size = (150, 150), batch_size = 30, class_mode = "binary", classes = ["cats", "dogs"])
      test_generator = test_datagen.flow_from_directory( self.test_dir,  target_size = (150, 150), batch_size = 20, class_mode = "binary", classes = ["cats", "dogs"] )
      return train_generator, test_generator

    def baseline_model( self ):
      model = models.Sequential()
      model.add( layers.Conv2D( 32, (3, 3), activation = "relu", input_shape = (150, 150, 3) ) )
      model.add( layers.MaxPooling2D(2, 2))
      model.add( layers.Flatten() )
      model.add( layers.Dense( 128, activation = "relu") )
      model.add( layers.Dense(1, activation = "sigmoid") )
      model.compile( loss = "binary_crossentropy", optimizer = optimizers.RMSprop( learning_rate = 0.0001 ) )
      history = model.fit( self.train_generator, steps_per_epoch = 50, epochs = 2, validation_data = self.test_generator, validation_steps = 50)
      return model 

obj = CatVsDogs()
#obj.visualize_pics("cat", 0, 9 ) #this works
#obj.make_folders() #this works
#obj.copy_images() #this works
#obj.check_data() #this works
obj.baseline_model()
