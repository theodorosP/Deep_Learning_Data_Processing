import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class CatVsDogs():
  #define constructor
  def __init__(self):
    self.folder = "/content/drive/MyDrive/dogs-vs-cats/train/train/"

  def view_as_subplot( self, name ):
    for i in range( 0, 9 ):
        plt.subplot( 330 + 1 + i )
        filename = self.folder + name.lower() + "." + str( i ) + ".jpg"
        image = mpimg.imread( filename )
        plt.imshow( image )
    plt.show()

  #name should be either cat or dog
  def visualize_pics( self, name ):
    try:
      if name.lower() == "cat":
        self.view_as_subplot( name.lower() )
      elif name.lower() == "dog":
        self.view_as_subplot ( name )
      else:
        raise ValueError( "The name should be either dog or cat" )
    except Exception as e:
      print("An error occured:", e)


obj = CatVsDogs()
obj.visualize_pics("cat")
