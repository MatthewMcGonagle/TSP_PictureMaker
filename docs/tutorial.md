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

![Original Tiger Head Resize](https://raw.githubusercontent.com/MatthewMcGonagle/TSP_PictureMaker/master/tigerHeadResize.png)

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

# Running the First Step of Annealing, the `sizeScale` Annealer

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

For the picture in the tutorial, we are able to get the energy down to about 129.98. We decrease sizeScale to
about 0.02 in decrements of 0.01. Each time we reset the temperature to 0.001. To know when to change the size 
scale, look for when there is very little improvement in the energy (keep an eye on the max and min of the 
y axis for the energy graph).

# Changing the Annealer for the Second Step

Now we are ready to change the annealer. You have the ability to change to any available annealer type, but we 
recommend changing to the `sizeNeighbors` annealer for the second step. This is the annealer the tutorial
will be switching to. To change the annealer, 
1. Press `m` for the menu.
2. Enter `a` for changing the annealer.
3. Enter `i` for the `sizeNeighbors` annealer.
4. Make sure the temperature looks reasonable compared to what you have been using before. It is possible
that the program has made a bad guess at a nice temperature for this annealer.
5. If you need to switch the temperature, then enter `t` to change the temperature and enter a new 
floating point temperature. We recommend `0.001`.
6. Enter `c` to continue.

# Using the `sizeNeighbors` Annealer

The `sizeNeighbors` annealer uses a minimum size scale to find a pool of vertices to choose from (similar
to the `sizeScale` annealer). However, after choosing one vertex from the pool, the second vertex is
now randomly chosen from the neighbors of that vertex. The idea is to concentrate on the largest edges and
try to decrease their size by switching place with nearby vertices.

As you run the annealer, you will have to change the size scale. To do so, the directions are exactly the
same for the `sizeScale` annealer described above. Similarly, you will need to reset the temperature.

You should run this annealer until the size of the pool `nPool` won't decrease beyond a reasonable size. 

For this tutorial example, you should have success by decreasing the size scale to 0.01 in decrements of 0.01.
Then try decreasing to a size scale of 0.005 in decrements that don't go too fast (maybe try 0.001 or 0.002).
Try to make sure there aren't too many points in the pool; so try to keep `nPool` below say 5000.
Also watch out for when you have diminishing returns on the energy decrease in the energy graph. Doing so we
are able to get the energy down to about 108.80. 


There isn't really any hard and fast numbers to make this work. You sort of need to get a feel for it and
just let the program run to do its job.

# Using the `neighbors` Annealer, Final Step

The `neighbors` annealer doesn't use a pool of vertices. It randomly chooses a first vertex from all of the
vertices, and then chooses a second vertex from its neighbors. To change to the `neigbors` annealer:
1. Press `m` for menu.
2. Enter `a` to change the annealer.
3. Enter `n` for the neighbors annealer.
4. Check your temperature, and change it if needed.
5. Enter `c` to continue.

After letting it run (and once in a while resetting the temperature to about 0.0007), we get an
energy of 108.03. Let's now graph and save the result.

# Graphing and Saving the Result

Follow these steps:
1. Press `m` for menu.
2. Press `r` for graph results. This opens an interactive `pyplot` window containing the graph of 
the cycle.
3. In the graph window, click on the disk icon to save a copy of the graph.
4. Save a version of the graph. We recommend saving it as a vector graphics `.svg` file so
that you can edit the graph using other software. 

If you save a `.svg` version of the file, then you can use vector graphics editing software such as
[Inkscape](https://inkscape.org/) to edit the image. For example, you can use Inkscape to remove
the axes (you will need to ungroup the objects in the image), change the background, or add
color gradients. 

# Exiting

To exit, from the menu, simply enter `s`.
