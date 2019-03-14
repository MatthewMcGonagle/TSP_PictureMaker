import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tsp_draw

###########################
#### Main executable
##########################

def main():

    # Set the figure size for graphs of cycles.

    cycleFigSize = (8, 8)
    finalFigSize = (10, 10)

    # Open the image.

    image = Image.open('tigerHeadResize.png').convert('L')
    pixels = tsp_draw.graphics.getPixels(image, ds = 1)
    plt.imshow(pixels, cmap = 'gray')
    plt.show()

    # Get the dithered image.

    ditheringMaker = tsp_draw.dithering.DitheringMaker()
    dithering = ditheringMaker.make_dithering(pixels)
    plt.imshow(dithering, cmap = 'gray')
    plt.show()

    # Get the vertices from the dithered image and then
    # do the preprocessing.
    
    vertices = tsp_draw.dithering.get_vertices(dithering)
    print('Num vertices = ', len(vertices))
    print('Preprocessing Vertices')
    vertices = tsp_draw.process_vertices.preprocess(vertices)
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
    
    annealingSteps = tsp_draw.size_scale.Annealer(nSteps / nJobs, vertices, initTemp, cooling, initScale, sizeCooling)
    print('Initial Configuration:\n', annealingSteps.get_info_string())
    
    # Plot the intial cycle.
    
    cycle = annealingSteps.get_cycle()
    tsp_draw.graphics.plotCycle(cycle, 'Greedy Guess Path', doScatter = False, figsize = cycleFigSize)
    plt.tight_layout()
    tsp_draw.graphics.savePNG('docs\\greedyGuess.png')
    plt.show()
    
    energies = tsp_draw.jobs.do_annealing(annealingSteps, nJobs)
 
    print('Finished running annealing jobs') 
    
    # Plot the energies of the annealing process over time.
    
    tsp_draw.graphics.plotEnergies(energies, 'Energies for Size Scale Annealing')
    # If you wish to save a copy of the graph, then use the following line:
    # tsp_draw.graphics.savePNG('docs\\sizeScaleEnergies.png')
    plt.show()
    
    # Plot the final cycle of the annealing process.
    
    cycle = annealingSteps.get_cycle()
    tsp_draw.graphics.plotCycle(cycle, 'Final Path for Size Scale Annealing', doScatter = False, figsize = cycleFigSize)
    plt.tight_layout()
    tsp_draw.graphics.savePNG('docs\\afterSizeAnnealing.png')
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
    
    annealingSteps = tsp_draw.neighbors.Annealer(nSteps / nJobs, vertices, initTemp, cooling, initNbrs, nbrsCooling)
    print('Initial Configuration:\n', annealingSteps.get_info_string())
    
    # Now run the annealing steps for the vonNeumann.png example.
   
    energies = tsp_draw.jobs.do_annealing(annealingSteps, nJobs) 
    print('Finished running annealing jobs') 
    
    # Plot the energies of the annealing process over time.
    
    tsp_draw.graphics.plotEnergies(energies, 'Energies for Neighbors Annealing')
    # If you wish to save a copy of this graph, then use the following line:
    # tsp_draw.graphics.savePNG('docs\\nbrsEnergies.png')
    plt.show()
    
    # Plot the final cycle of the annealing process.
    
    cycle = annealingSteps.get_cycle()
    tsp_draw.graphics.plotCycle(cycle, 'Final Path for Neighbors Annealing', doScatter = False, figsize = finalFigSize)
    plt.tight_layout()
    tsp_draw.graphics.savePNG('docs\\finalCycle.png')
    plt.show()   

##########################
#### The actual execution 
            
main()  
