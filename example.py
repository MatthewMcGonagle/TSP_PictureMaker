import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from PIL import Image
from annealers import *
from dithering import *
from processVertices import *
from resultPlotting import *

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
 
###########################
#### Main executable
##########################

def main():

    # Set the figure size for graphs of cycles.

    cycleFigSize = (8, 8)
    finalFigSize = (10, 10)

    # Open the image.

    image = Image.open('tigerHeadResize.png').convert('L')
    pixels = getPixels(image, ds = 1)
    plt.imshow(pixels, cmap = 'gray')
    plt.show()

    # Get the dithered image.

    ditheringMaker = DitheringMaker()
    dithering = ditheringMaker.makeDithering(pixels)
    plt.imshow(dithering, cmap = 'gray')
    plt.show()

    # Get the vertices from the dithered image and then
    # do the preprocessing.
    
    vertices = getVertices(dithering)
    print('Num vertices = ', len(vertices))
    print('Preprocessing Vertices')
    vertices = preprocessVertices(vertices)
    print('Preprocessing Complete')
    plt.scatter(vertices[:, 0], vertices[:, 1])
    plt.show()

    ######################################
    ############# Annealing based on size
    ######################################

    # Set up parameters for annealing based on size scale.
    nVert = len(vertices) 
    initTemp = 0.1 * 1 / np.sqrt(nVert) / np.sqrt(np.pi) 
    nSteps = 5 * 10**5 
    decimalCool = 1.5
    cooling = np.exp(-np.log(10) * decimalCool / nSteps) 
    nJobs = 200

    # Size scale parameters are based on statistics of sizes of current edges.

    distances = np.linalg.norm(vertices[1:] - vertices[:-1], axis = -1)
    initScale = np.percentile(distances, 99.9)
    finalScale = np.percentile(distances, 90.8)
    sizeCooling = np.exp(np.log(finalScale / initScale) /nSteps)
    
    # Set up our annealing steps iterator.
    
    annealingSteps = AnnealerTSPSizeScale(nSteps / nJobs, vertices, initTemp, cooling, initScale, sizeCooling)
    print('Initial Configuration:\n', annealingSteps.getInfoString())
    
    # Plot the intial cycle.
    
    cycle = annealingSteps.getCycle()
    plotCycle(cycle, 'Greedy Guess Path', doScatter = False, figsize = cycleFigSize)
    plt.tight_layout()
    savePNG('docs\\greedyGuess.png')
    plt.show()
    
    energies = doAnnealingJobs(annealingSteps, nJobs)
 
    print('Finished running annealing jobs') 
    
    # Plot the energies of the annealing process over time.
    
    plotEnergies(energies, 'Energies for Size Scale Annealing')
    # If you wish to save a copy of the graph, then use the following line:
    # savePNG('docs\\sizeScaleEnergies.png')
    plt.show()
    
    # Plot the final cycle of the annealing process.
    
    cycle = annealingSteps.getCycle()
    plotCycle(cycle, 'Final Path for Size Scale Annealing', doScatter = False, figsize = cycleFigSize)
    plt.tight_layout()
    savePNG('docs\\afterSizeAnnealing.png')
    plt.show()   

    vertices = cycle[:-1]
    print('Double check: num vertices = ', len(vertices))

    #################################
    ### Now do TSP based on neighbors
    #################################
       
    # Set up parameters for annealing based on neighbors.
 
    nVert = len(vertices) 
    initTemp = 0.1 * 1 / np.sqrt(nVert) / np.sqrt(np.pi) 
    nSteps = 5 * 10**5 
    decimalCool = 1.5
    cooling = np.exp(-np.log(10) * decimalCool / nSteps) 
    nJobs = 200

    # Neighbor parameters are set up by trial and error.

    initNbrs = int(nVert * 0.01)
    initNbrs = 50
    finalNbrs = 3 
    nbrsCooling = np.exp(np.log(finalNbrs / initNbrs) / nSteps) 
    
    # Set up our annealing steps iterator.
    
    annealingSteps = NeighborsAnnealer(nSteps / nJobs, vertices, initTemp, cooling, initNbrs, nbrsCooling)
    print('Initial Configuration:\n', annealingSteps.getInfoString())
    
    # Now run the annealing steps for the vonNeumann.png example.
   
    energies = doAnnealingJobs(annealingSteps, nJobs) 
    print('Finished running annealing jobs') 
    
    # Plot the energies of the annealing process over time.
    
    plotEnergies(energies, 'Energies for Neighbors Annealing')
    # If you wish to save a copy of this graph, then use the following line:
    # savePNG('docs\\nbrsEnergies.png')
    plt.show()
    
    # Plot the final cycle of the annealing process.
    
    cycle = annealingSteps.getCycle()
    plotCycle(cycle, 'Final Path for Neighbors Annealing', doScatter = False, figsize = finalFigSize)
    plt.tight_layout()
    savePNG('docs\\finalCycle.png')
    plt.show()   

##########################
#### The actual execution 
            
main()  
