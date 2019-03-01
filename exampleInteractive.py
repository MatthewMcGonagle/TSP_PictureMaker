import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tspDraw

myInputFileName = 'tigerHeadResize.png'

###########################
#### Main executable
##########################


# Set the figure size for graphs of cycles.

cycleFigSize = (8, 8)
finalFigSize = (10, 10)

# Open the image.

image = Image.open(myInputFileName).convert('L')
pixels = tspDraw.graphics.getPixels(image, ds = 1)
plt.imshow(pixels, cmap = 'gray')
plt.show()

# Get the dithered image and its vertices.

ditheringMaker = tspDraw.dithering.DitheringMaker()
dithering = ditheringMaker.makeDithering(pixels)
vertices = tspDraw.dithering.getVertices(dithering)

print("Number Vertices = ", len(vertices))
plt.imshow(dithering, cmap = 'gray')
plt.show()

# Do the preprocessing of the vertices.

vertices = tspDraw.processVertices.preprocess(vertices)
print('Preprocessing Complete')

session = tspDraw.interactive.Session(vertices)
session.run()
