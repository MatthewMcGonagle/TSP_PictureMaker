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

# Get the dithered image.

ditheringMaker = tspDraw.dithering.DitheringMaker()
dithering = ditheringMaker.makeDithering(pixels)
plt.imshow(dithering, cmap = 'gray')
plt.show()

# Get the vertices from the dithered image and then
# do the preprocessing.

vertices = tspDraw.dithering.getVertices(dithering)
print("Number Vertices = ", len(vertices))
vertices = tspDraw.processVertices.preprocess(vertices)
print('Preprocessing Complete')
plt.scatter(vertices[:, 0], vertices[:, 1])
plt.show()

session = tspDraw.interactive.Session(vertices)
session.run()
