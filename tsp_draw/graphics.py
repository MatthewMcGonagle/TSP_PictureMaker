import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

def getPixels(image, ds = 1):
    '''
    Get the pixels as a numpy array from a PIL image.
    We can take the mean of each ds x ds subsquare as an array element inorder to down-size
    the size of the image if we want to.

    Parameters
    ----------
    image : PIL Image
        The PIL image to convert.

    ds : Int
        We take the mean of each ds x ds sub-square for a single element of our array. 

    Returns
    -------
    2d Numpy array of floats
        The converted values of the pixels in the image. We use mean because we
        possibly took a mean over sub-squares.
    '''

    imwidth, imheight = image.size
    pixels = list(image.getdata())
    pixels = np.array(pixels).reshape((imheight, imwidth))
    
    pixels = [[pixels[i:i+ds, j:j+ds].mean() for j in np.arange(0, imwidth, ds)] 
                for i in np.arange(0, imheight, ds)]
    pixels = np.array(pixels)
    return pixels
 
def plotCycle(cycle, title, doScatter = True, figsize = (5, 5)):
    ''' 
    Plot a cycle through all of the vertices. Can optionally do a scatter plot of all the vertices.

    Parameters
    ----------
    cycle : numpy array of floats of shape (nVertices, 2)
        Holds the vertices of the cycle. The endpoints should be the same to make a closed cycle,
        i.e. cycle[0] == cycle[nVertices - 1].

    title : String
        The title of the graph.

    doScatter : Boolean
        Whether to do a scatter plot of the different vertices. Default is True, i.e. do the
        scatter plot.

    figsize : Pair of Int
        The size of the figure to use for drawing the cycle. Default is (5, 5).
    '''

    plt.figure(figsize = figsize) 
    plt.plot(cycle[:, 0], cycle[:, 1])
    if doScatter:
        plt.scatter(cycle[:, 0], cycle[:, 1], color = 'red')
    ax = plt.gca()
    ax.set_title(title)

def plotEnergies(energies, title):
    '''
    Plot energies collected while running the annealing algorithm.

    Parameters
    ----------
    energies : numpy array of shape (Num Energies)
        The energies to graph.

    title : String
        The title of the graph.
    '''

    plt.figure(figsize = (5, 5))
    plt.plot(np.log(energies) / np.log(10))
    ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel('Nth Run of Annealing')
    ax.set_ylabel('Energy')

def savePNG(filename):
    '''
    Save the current pyplot figure as a png file, but first reduce the color palette to reduce the file size.

    Parameters
    ----------
    filename : String
        The name of the file to save to.
    '''

    imageIO = io.BytesIO()
    plt.savefig(imageIO, format = 'png')
    imageIO.seek(0)
    image = Image.open(imageIO)
    image = image.convert('P', palette = Image.WEB)
    image.save(filename , format = 'PNG')
