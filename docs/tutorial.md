---
layout: default
title: Using the Interactive Session Tutorial
---

Here we will go over how to use the interactive session provided by `exampleInteractive.py` to use
`TSP_PictureMaker`. Now, please note that `TSP_PictureMaker` uses random behavior when searching for
approximate solutions to the Travelling Salesman Problem. So your results will NOT exactly match
those given in the tutorial. However, they should be similar.

# Picking a Picture

For this tutorial, we will be using `tigerHeadResize.png` as found in the root directory of the project: 
![Original Tiger Head Resize](../tigerHeadResize.png)

Now, for finding your own pictures, there are a couple things you will want to take into account.
1. Don't use a picture that is too dark. The darker your picture, the more vertices that will be
generated for the Travelling Salesman Problem. Too many vertices will slow down the algorithm (in 
a nonlinear way).
2. Don't use a picture that is too large. Larger pictures will generate more vertices. 

To remedy any of these issues, you can use a image editing software, such as 
[GNU Image Manipulation Program](https://www.gimp.org/). It is helpful to use such software
in any of the following ways:
1. Use smart scissors to remove the part of the picture you are interested in from the background. 
2. Adjust the color levels (in particular the `value` levels) using curves.
3. Try inverting colors to turn darks to whites (and vice versa).
4. Scale the size of the picture. The number of vertices should approximately change with the square of
the scaling. For example, if you half the width and height of the picture, then the number of vertices
should be a quarter of what you had before.

Finally, your picture should be in a standard format (such as '.png'). When using image editing software,
make sure you export the image to a standard format.

# Adding the Picture to the Session

Currently the filename has to be manually added to `exampleInteractive.py`; however, this is a simple change.
Just change the value of `myInputFileName` in `exampleInteractive.py` to the name of the picture file 
you wish to use. For example, for this tutorial, it is set to
``` python
myInputFileName = 'tigerHeadResize.png'
```

# Starting the Interactive Session

To start the interactive session, simply run 
```
python exampleInteractive.py
```
from a terminal at the root directory of the project.

# Checking out the Picture and Vertices

First, a greyscale version of the picture will open in a window. If it is okay, just close the window. If not,
then terminate the application.

Next, a black and white picture of the vertices will open in a window; also the number of vertices generated
will be printed on the terminal. For our tutorial, you should see
```
Number Vertices =  22694
``` 
If the number of vertices is good, then close the picture window; if not, then close the application and make
appropriate changes to the picture using image editing software.

# PreProcess the Vertices

The program will now run some preprocessing on the vertices. This includes:
1. Scaling the vertices to make everything a more uniform size.
2. Making a greedy initial guess for the solution. This may discard one of the vertices (which shouldn't
matter much if we have a lot of vertices).

The greedy guess may take some time to make. Please be patient as it is being created. Once it is done, the
interactive annealing will start.

# Running the First Step of Annealing

There are different types of annealers to run on the problem. The interactive session starts with `sizeScale`
annealer. This annealer starts by making a pool of vertices, such that each vertex touches an edge that is
at least as large as the current choice of size scale. As the annealer runs, it only looks at switching the order
of vertices that are in the pool. The idea is that the annealer can focus on only reducing the size 
of the largest edges in the cycle.

The pool is recreated by periodically doing a warm restart of the annealer. As the annealer reduces lengths
of edges in the cycle, the number of vertices in the pool will decrease. To put more vertices in the pool,
you need to decrease the size scale:
1. Press `m` for menu.
2. Enter `e` for change scale.
3. Enter a floating point size. Look at the current value of `sizeScale` as displayed on the terminal
to get an idea of what to enter.
4. On the terminal, look at the size of `nPool` the number of vertices in the pool. If it is too small or too
large, then repeat to select a different size.
5. Enter `c` for continue.

Note, if at any point the pool has less than two vertices, the execution of the annealing will stop, and the 
program will ask you to change the size scale until a pool with enough vertices is created. 

The temperature of the annealer automatically cools as it is run. You may find that you want to reset it to
a higher temperature as you are running it. To do so simply
1. Press `m` for menu.
2. Enter `t` for change temperature.
3. Enter a floating point for the temperature. Be careful, putting in a temperature that is too high could
result in undoing work that has already been done.
4. Enter `c` for continue.

To see what the current cycle looks like simply:
1. Press `m` for menu.
2. Enter `r` for graphing the result.
3. Look at the picture window to see what the current cycle looks like. When you are done, close the picture
window, and the terminal will return to the main menu.
4. Enter `c` for continue. 
