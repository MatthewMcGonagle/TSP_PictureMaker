import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tsp_draw

myInputFileName = 'tigerHeadResize.png'

###########################
#### Main executable
##########################


# Set the figure size for graphs of cycles.

cycleFigSize = (8, 8)
finalFigSize = (10, 10)

# Open the image.

image = Image.open(myInputFileName).convert('L')
pixels = tsp_draw.graphics.getPixels(image, ds = 1)
plt.imshow(pixels, cmap = 'gray')
plt.show()

# Get the dithered image and its vertices.

ditheringMaker = tsp_draw.dithering.DitheringMaker()
dithering = ditheringMaker.make_dithering(pixels)
vertices = tsp_draw.dithering.get_vertices(dithering)

print("Number Vertices = ", len(vertices))
plt.imshow(dithering, cmap = 'gray')
plt.show()

# Do the preprocessing of the vertices.

vertices = tsp_draw.process_vertices.preprocess(vertices)
print('Preprocessing Complete')

session = tsp_draw.interactive.Session(vertices)
session.run()
