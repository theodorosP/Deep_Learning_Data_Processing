import re
import os
import shutil
import random as random
from keras import optimizers
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.callbacks import EarlyStopping
from keras import layers, Sequential, models
from keras.preprocessing.image import ImageDataGenerator

class CatVsDogs():
    #define constructor
    def __init__(self):
        self.train_dir = "/content/drive/MyDrive/dogs-vs-cats/train/"
        self.test_dir = "/content/drive/MyDrive/dogs-vs-cats/test/"
        self.validation_dir = "/content/drive/MyDrive/dogs-vs-cats/validation/"
        self.train_generator, self.test_generator, self.validation_generator = self.generators()
        self.data_dir = "/content/drive/MyDrive/dogs-vs-cats/data/"
        self.cat_dog_dir = "/content/drive/MyDrive/dogs-vs-cats/"
        self.val_ratio = 0.2
        self.test_ratio = 0.25

    #name is either cat or dog
    #start_index and end_index are the indexes of picures we want to visualize
    def view_as_subplot( self, name, start_index, end_index ):
        try:
          if abs( start_index - end_index ) <= 9:
            for i in range( start_index, end_index ):
              plt.subplot( 330  + i - start_index + 1  )
              filename = self.data_dir + name.lower() + "." + str( i ) + ".jpg"
              print("Reading file:", filename )
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
      train_test_validation = [ "train/", "test/", "validation/" ]
      dogs_cats = [ "dogs/", "cats/" ]
      for i in train_test_validation:
        for j in dogs_cats:
          new_dir = self.cat_dog_dir + i + j
          if os.path.exists( new_dir ):
            print( "Folder exists", new_dir )
          else:
            os.makedirs( new_dir )

    def get_training_test_validation_lists( self, train_test_or_validation ):
      try:
        if train_test_or_validation.lower() == "train_test":
          files = os.listdir( self.data_dir )
          data_length = int( len(files) / 2 ) #dogs and cats are the same, so I divide by 2
          test_images = list()
          random.seed( 1 )
          for i in range( 0, data_length ):
            random_number = random.random()
            if random_number <= self.test_ratio:
              test_images.append( i )
          train_images = [ i for i in range(0, data_length) if i not in test_images ]
          return train_images, test_images
        elif train_test_or_validation.lower() == "validation":
          files = os.listdir( self.train_dir  + "dogs" )
          validation_images = list()
          random.seed( 1 )
          for i in files:
            i = re.sub(r'\D', '', i )
            random_number = random.random()
            if random_number < self.val_ratio:
              validation_images.append( int( i ) )
          return sorted( validation_images )
        else:
           raise ValueError("The train_test_or_validation can take values train_test, if you are attempting to coppy train_test data or validation if you are trying to copy validation data!")
      except Exception as e:
        print("An error occured:", e)

    def copy_to_path( self, image_index_list, train_or_test_or_validation ):
      try:
        if train_or_test_or_validation.lower() == "test":
          dogs_destination = self.test_dir + "dogs/"
          cats_destination = self.test_dir + "cats/"
        elif train_or_test_or_validation.lower() == "train":
          dogs_destination = self.train_dir + "dogs/"
          cats_destination = self.train_dir + "cats/"
        elif train_or_test_or_validation.lower() == "validation":
          dogs_destination = self.validation_dir + "dogs/"
          cats_destination = self.validation_dir + "cats/"
        else:
          raise ValueError("The train_or_test_or_validation should be either train or test or validation.")
      except Exception as e:
        print("An error occured:", e)
      for i in image_index_list:
        dog_image = "dog." + str( i ) + ".jpg"
        cat_image = "cat." + str( i ) + ".jpg"
        for j in [ dog_image, cat_image ]:
          if j == dog_image and os.path.exists( dogs_destination + j ):
            print( "File:",  dogs_destination + j, "exists" )
          elif j == dog_image and os.path.exists( dogs_destination + j ) == False:
            print("File:",  dogs_destination +  j, " NOT exists" )
            os.system("cp " +  self.data_dir + dog_image + " " + dogs_destination)
            print( dog_image, "coppied to:", dogs_destination )
          elif j == cat_image and os.path.exists( cats_destination + j ):
            print( "File:",  cats_destination + j, "exists" )
          elif j == cat_image and os.path.exists( cats_destination + j ) == False :
             os.system("cp " +  self.data_dir + cat_image + " " + cats_destination)
             print( cat_image, "coppied to:", cats_destination )

    def copy_images_to_train_test_validation( self ):
      train_images, test_images = self.get_training_test_validation_lists( "train_test" )
      self.copy_to_path( test_images, "test" ) #keep the order training directory has to be completed before moving to validation
      self.copy_to_path( train_images, "train" ) #keep the order
      validation_images = self.get_training_test_validation_lists( "validation" )
      self.copy_to_path( validation_images, "validation" )
      self.remove_dublicates_from_training( validation_images )


    def remove_dublicates_from_training( self, validation_images ):
       for i in validation_images:
         for j in ["cats/", "dogs/"]:
           try:
              if os.path.exists( self.train_dir + j + j[ : -2] + "." + str( i )  + ".jpg" ):
                print("file:", self.train_dir + j + j[ : -2] + "." + str( i )  + ".jpg", "exists")
                os.remove( self.train_dir + j + j[ : -2] + "." + str( i )  + ".jpg" )
                print("file", self.train_dir + j + j[ : -2] + "." + str( i )  + ".jpg", "sucesfully deleted from training dataset")
           except:
             print("Error while deleting file : ", self.train_dir + j + j[ : -2] + "." + str( i )  + ".jpg" )


    def check_data( self ):
      for i in ["train", "test", "validation"]:
        for j in ["cats", "dogs"]:
          print("----" * 10)
          files = os.listdir( "/content/drive/MyDrive/dogs-vs-cats/" + i + "/" + j )
          print(i, j, len( files ))

    def generators( self ):
      train_datagen = ImageDataGenerator( rescale = 1./255 )
      test_datagen = ImageDataGenerator( rescale = 1./255 )
      validation_datagen = ImageDataGenerator( rescale = 1./255 )
      train_generator = train_datagen.flow_from_directory( self.train_dir, target_size = (150, 150), batch_size = 20, class_mode = "binary", classes = ["cats", "dogs"] )
      test_generator = test_datagen.flow_from_directory( self.test_dir, target_size = (150, 150), batch_size = 20, class_mode = "binary", classes = ["cats", "dogs"] )
      validation_generator = validation_datagen.flow_from_directory( self.validation_dir,  target_size = (150, 150), batch_size = 20, class_mode = "binary", classes = ["cats", "dogs"] )
      return train_generator, test_generator, validation_generator

    def plot( self, history, key_word ):
      try:
        if key_word.lower() == "loss":
          plt.plot( range( 1, len( history.history[ "loss" ] ) + 1 ),  history.history[ "loss" ], "-o", label = "Loss" )
          plt.plot( range( 1, len( history.history[ "val_loss" ] ) + 1 ),  history.history[ "val_loss" ], "-o", label = "Val_Loss" )
          plt.xlabel( "epochs" )
          plt.ylabel( "Loss" )
          plt.legend( loc = "best" )
          plt.show()
        elif key_word.lower() == "accuracy":
          plt.plot( range( 1, len( history.history[ "accuracy" ] ) + 1 ),  history.history[ "accuracy" ], "-o", label = "Accuracy" )
          plt.plot( range( 1, len( history.history[ "val_accuracy" ] ) + 1 ),  history.history[ "val_accuracy" ], "-o", label = "Val_Accuracy" )
          plt.xlabel( "epochs" )
          plt.ylabel( "Accuracy" )
          plt.legend( loc = "best" )
          plt.show()
        else:
          raise ValueError("The key word can either be loss or accuracy.")
      except exception as e:
        print("An error occured:", e)

    def print_and_plot( self, history, results):
      print("history =", history )
      print("resukts =", results )
      loss, accuracy = results[ 0 ], results[ 1 ]
      ratio = round( results[ 0 ] / history.history[ "loss" ][ -1 ], 2 )
      print( "test_loss/training_loss = ",  ratio )
      for i in [ "loss", "accuracy" ]:
        self.plot( history, i )

    def baseline_model( self ):
      random.seed(1)
      model = models.Sequential()
      model.add( layers.Conv2D( 32, ( 3, 3 ), activation = "relu", input_shape = ( 150, 150, 3 ) ) )
      model.add( layers.MaxPooling2D( 2, 2 ))
      model.add( layers.Flatten() )
      model.add( layers.Dense( 128, activation = "relu") )
      model.add( layers.Dense( 1, activation = "sigmoid") )
      model.compile( loss = "binary_crossentropy", optimizer = optimizers.RMSprop( learning_rate = 0.0001 ), metrics = [ "accuracy" ] )
      early_stopping = EarlyStopping( monitor='val_loss', patience = 3 , restore_best_weights = True )
      history = model.fit( self.train_generator, batch_size = 150, steps_per_epoch = 50, epochs = 40, validation_data = self.validation_generator, validation_steps = 50, callbacks = [early_stopping])
      results = model.evaluate( self.test_generator )
      self.print_and_plot( history, results )
      return model, history, results

    def second_model( self ):
      random.seed(1)
      model = models.Sequential()
      model.add( layers.Conv2D( 32, ( 3, 3 ), activation = "relu", input_shape = (150, 150, 3 ) ) )
      model.add( layers.MaxPooling2D( 2, 2 ) )
      model.add( layers.Conv2D( 64, (3, 3), activation = "relu") )
      model.add( layers.MaxPooling2D( 2, 2)  )
      model.add( layers.Flatten() )
      model.add(layers.Dense( 128, activation = "relu") )
      model.add( layers.Dense( 1, activation = "sigmoid" ) )
      model.compile( loss = "binary_crossentropy", optimizer =  optimizers.RMSprop( learning_rate = 0.0001 ), metrics = ["accuracy"] )
      early_stopping = EarlyStopping( monitor='val_loss', patience = 3 , restore_best_weights = True )
      history = model.fit( self.train_generator, batch_size = 150, steps_per_epoch = 50, epochs = 20, validation_data = self.validation_generator, validation_steps = 50, callbacks = early_stopping )
      results = model.evaluate( self.test_generator )
      self.print_and_plot( history, results )
      return model, history, results

    def third_model( self ):
      random.seed(1)
      model = models.Sequential()
      model.add( layers.Conv2D( 32, ( 3, 3 ), activation = "relu", input_shape = (150, 150, 3 ) ) )
      model.add( layers.MaxPooling2D( 2, 2 ) )
      model.add( layers.Conv2D( 64, (3, 3), activation = "relu") )
      model.add( layers.MaxPooling2D( 2, 2)  )
      model.add( layers.Conv2D( 128, (3, 3), activation = "relu") )
      model.add( layers.MaxPooling2D( 2, 2)  )
      model.add( layers.Flatten() )
      model.add(layers.Dense( 128, activation = "relu") )
      model.add( layers.Dense( 1, activation = "sigmoid" ) )
      model.compile( loss = "binary_crossentropy", optimizer =  optimizers.RMSprop( learning_rate = 0.0001 ), metrics = ["accuracy"] )
      early_stopping = EarlyStopping( monitor='val_loss', patience = 3 , restore_best_weights = True )
      history = model.fit( self.train_generator, batch_size = 150, steps_per_epoch = 50, epochs = 20, validation_data = self.validation_generator, validation_steps = 50, callbacks = early_stopping )
      results = model.evaluate( self.test_generator )
      self.print_and_plot( history, results )
      return model, history, results

    def fourth_model( self ):
      random.seed(1)
      model = models.Sequential()
      model.add( layers.Conv2D( 32, ( 3, 3 ), activation = "relu", input_shape = (150, 150, 3 ) ) )
      model.add( layers.MaxPooling2D( 2, 2 ) )
      model.add( layers.Conv2D( 64, (3, 3), activation = "relu") )
      model.add( layers.MaxPooling2D( 2, 2)  )
      model.add( layers.Conv2D( 128, (3, 3), activation = "relu") )
      model.add( layers.MaxPooling2D( 2, 2)  )
      model.add( layers.Flatten() )
      model.add(layers.Dense( 512, activation = "relu") ) #increase from 128
      model.add( layers.Dense( 1, activation = "sigmoid" ) )
      model.compile( loss = "binary_crossentropy", optimizer =  optimizers.RMSprop( learning_rate = 0.0001 ), metrics = ["accuracy"] )
      early_stopping = EarlyStopping( monitor='val_loss', patience = 3 , restore_best_weights = True )
      history = model.fit( self.train_generator, batch_size = 150, steps_per_epoch = 50, epochs = 20, validation_data = self.validation_generator, validation_steps = 50, callbacks = early_stopping )
      results = model.evaluate( self.test_generator )
      self.print_and_plot( history, results )
      return model, history, results

    def fifth_model( self ):
      random.seed(1)
      model = models.Sequential()
      model.add( layers.Conv2D( 32, ( 3, 3 ), activation = "relu", input_shape = (150, 150, 3 ) ) )
      model.add( layers.MaxPooling2D( 2, 2 ) )
      model.add( layers.Conv2D( 64, (3, 3), activation = "relu") )
      model.add( layers.MaxPooling2D( 2, 2)  )
      model.add( layers.Conv2D( 128, (3, 3), activation = "relu") )
      model.add( layers.MaxPooling2D( 2, 2)  )
      model.add( layers.Flatten() )
      model.add(layers.Dense( 512, activation = "relu") )
      model.add(layers.Dense( 128, activation = "relu") )
      model.add( layers.Dense( 1, activation = "sigmoid" ) )
      model.compile( loss = "binary_crossentropy", optimizer =  optimizers.RMSprop( learning_rate = 0.0001 ), metrics = ["accuracy"] )
      early_stopping = EarlyStopping( monitor='val_loss', patience = 3 , restore_best_weights = True )
      history = model.fit( self.train_generator, batch_size = 150, steps_per_epoch = 50, epochs = 20, validation_data = self.validation_generator, validation_steps = 50, callbacks = early_stopping )
      results = model.evaluate( self.test_generator )
      self.print_and_plot( history, results )
      return model, history, results

    def sixth_model( self ):
      random.seed(1)
      model = models.Sequential()
      model.add( layers.Conv2D( 32, ( 3, 3 ), activation = "relu", input_shape = (150, 150, 3 ) ) )
      model.add( layers.MaxPooling2D( 2, 2 ) )
      model.add( layers.Conv2D( 64, (3, 3), activation = "relu") )
      model.add( layers.MaxPooling2D( 2, 2)  )
      model.add( layers.Conv2D( 128, (3, 3), activation = "relu") )
      model.add( layers.MaxPooling2D( 2, 2)  )
      model.add( layers.Flatten() )
      model.add( layers.Dropout( 0.5 ) )
      model.add(layers.Dense( 512, activation = "relu") )
      model.add(layers.Dense( 128, activation = "relu") )
      model.add( layers.Dropout( 0.5 ) )
      model.add( layers.Dense( 1, activation = "sigmoid" ) )
      model.compile( loss = "binary_crossentropy", optimizer =  optimizers.RMSprop( learning_rate = 0.0001 ), metrics = ["accuracy"] )
      early_stopping = EarlyStopping( monitor='val_loss', patience = 3 , restore_best_weights = True )
      history = model.fit( self.train_generator, batch_size = 150, steps_per_epoch = 50, epochs = 20, validation_data = self.validation_generator, validation_steps = 50, callbacks = early_stopping )
      results = model.evaluate( self.test_generator )
      self.print_and_plot( history, results )
      return model, history, results

    def seventh_model( self ):
      random.seed(1)
      model = models.Sequential()
      model.add( layers.Conv2D( 32, ( 3, 3 ), activation = "relu", input_shape = (150, 150, 3 ) ) )
      model.add( layers.MaxPooling2D( 2, 2 ) )
      model.add( layers.Conv2D( 64, (3, 3), activation = "relu") )
      model.add( layers.MaxPooling2D( 2, 2)  )
      model.add( layers.Conv2D( 128, (3, 3), activation = "relu") )
      model.add( layers.MaxPooling2D( 2, 2)  )
      model.add( layers.Flatten() )
      model.add( layers.Dropout( 0.5 ) )
      model.add(layers.Dense( 512, activation = "relu") )
      model.add( layers.Dropout( 0.5 ) )
      model.add( layers.Dense( 1, activation = "sigmoid" ) )
      model.compile( loss = "binary_crossentropy", optimizer =  optimizers.RMSprop( learning_rate = 0.0001 ), metrics = ["accuracy"] )
      early_stopping = EarlyStopping( monitor='val_loss', patience = 3 , restore_best_weights = True )
      history = model.fit( self.train_generator, batch_size = 150, steps_per_epoch = 50, epochs = 20, validation_data = self.validation_generator, validation_steps = 50, callbacks = early_stopping )
      results = model.evaluate( self.test_generator )
      self.print_and_plot( history, results )
      return model, history, results

    def eighth_model( self ):
      random.seed(1)
      model = models.Sequential()
      model.add( layers.Conv2D( 32, ( 3, 3 ), activation = "relu", input_shape = (150, 150, 3 ) ) )
      model.add( layers.MaxPooling2D( 2, 2 ) )
      model.add( layers.Conv2D( 64, (3, 3), activation = "relu") )
      model.add( layers.MaxPooling2D( 2, 2)  )
      model.add( layers.Conv2D( 128, (3, 3), activation = "relu") )
      model.add( layers.MaxPooling2D( 2, 2)  )
      model.add( layers.Flatten() )
      model.add( layers.Dropout( 0.5 ) )
      model.add(layers.Dense( 512, activation = "relu") )
      model.add(layers.Dense( 128, activation = "relu") )
      model.add( layers.Dropout( 0.5 ) )
      model.add( layers.Dense( 1, activation = "sigmoid" ) )
      model.compile( loss = "binary_crossentropy", optimizer =  optimizers.RMSprop( learning_rate = 0.0001 ), metrics = ["accuracy"] )
      early_stopping = EarlyStopping( monitor='val_loss', patience = 3 , restore_best_weights = True )
      history = model.fit( self.train_generator, batch_size = 150, steps_per_epoch = 50, epochs = 20, validation_data = self.validation_generator, validation_steps = 50, callbacks = early_stopping )
      results = model.evaluate( self.test_generator )
      self.print_and_plot( history, results )
      return model, history, results

obj = CatVsDogs()
#for i in ["cat", "dog"]:
#  obj.visualize_pics( i, 0, 9  ) #this works
#obj.make_folders() #this works
###a, b = obj.get_training_test_validation_lists( "train_test" ) #this works
###c = obj.get_training_test_validation_lists( "validation" ) #this works
#obj.copy_images_to_train_test_validation() #this works
#obj.check_data() #this works
model, history, results = obj.baseline_model() #this works
model, history, results = obj.second_model()
model, history, results = obj.third_model()
model, history, results = obj.fourth_model()
model, history, results = obj.fifth_model()
model, history, results = obj.sixth_model()
model, history, results = obj.seventh_model()
model, history, results = obj.eighth_model()
