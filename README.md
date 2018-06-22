`TSP_PictureMaker` is a module library for converting a picture to a a depiction created by an approximate solution
to the Travelling Salesman Problem (TSP). As an example, the code in `example.py` takes the following picture of a tiger's head:

![Original Tiger Head](tigerHeadResize.png)

and converts it to the following picture

![Approximate Solution to TSP for Tiger Head](docs\finalCycle.png)

The process to make the final picture are as follows:
1. The vertices for the TSP are created by dithering the input image and using the result black pixels as vertices.
2. An intial greedy guess for the solution to the TSP is made.
3. The greedy guess is improved by applying a modification of simulated annealing that is based on a decaying 
size scale.
4. Further improvements are made by applying another modification of simulated annealing based on nearest neighbors. 

# File Descriptions

## annealers.py

Classes responsible for doing simulated annealing based on size scale and for doing simulated annealing
based on k-nearest neighbors.

## dithering.py

Class and functions for applying dithering to an image.

## example.py

An example of using the functions to convert tigerHeadResize.png into a TSP picture.

## processVertices.py

Functions for getting the vertices from the dithered image and for preprocessing them, including the
initial greedy guess. 

## resultPlotting.py

Functions to make plotting results convenient.

# Description of Library's Process
