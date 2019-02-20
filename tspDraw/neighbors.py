import numpy as np
from sklearn.neighbors import NearestNeighbors
import tspDraw.base

##########################################
#### NeighborsAnnealer
##########################################

class Annealer( tspDraw.base.Annealer):
    '''
    Modified simulated annealer that randomly selects a vertex and then randomly selects another vertex from the
    k-nearest neighbors of the first vertex. The point of this annealer is to do annealing when the current
    cycle is at a stage where the changes needed to be made are by switching vertices that are close ot each other.    

    Members
    -------
    Members inherited from tspDraw.base.Annealer
    
    kNbrs : Float 
        The number of neighbors to randomly select from. This is converted to an int when doing selection. It is
        a float because we do geometric cooling on the number of neighbors as we run each step. 

    nbrsCooling : Float
        The factor to use to cool (decay) the number of neigbors. At each step, it is applied to kNbrs via 
        multiplication.        

    _nearestNbrs : class NearestNeighbors
        We use sci-kit-learn NearestNeighbors to find the nearest neighbors of vertices. This is trained on the
        original order of the vertices, so we need to deal with converting between the original order
        of the vertices to the current order of the vertices in the array.

    _origToCurrent : Numpy Array of Int of Shape (nVertices)
        Array for converting from original indices to current indices in cycle. That is
        origToCurrent[i] is the current index of what was originally the ith vertex. This
        is needed for dealing with results of nearest neighbors search. 

    _currentToOrig: Numpy Array of Int of Shape (nVertices)
        Array for converting from the current index of a vertex to the original index of the vertex.
        That is currentToOrig[i] is the original index of what is now index i in the cycle. This
        is needed to update origToCurrent when doing a reversal. 
    '''

    def __init__(self, nSteps, vertices, temperature, tempCool, kNbrs, nbrsCool):
        '''
        Initializer. Make sure to train the nearest neighbors on the original order of the vertices.

        Parameters
        ----------

        nSteps : Int
            The total number of iterations to make.

        vertices : Numpy array of Floats of shape (nVertices, 2)
            The vertices in their initial order.

        temperature : Float
            The initial temperature to use for the annealing.

        tempCool : Float
            The cooling factor to apply to the temperature at each step; it is applied
            via multiplication. That is, we have geometric cooling. 

        kNbrs : Float
            The initial value for kNbrs.

        nbrsCool : Float
            The cooling factor (decay factor) for the number of neighbors; at each step it
            is applied to kNbrs via multiplication. Note that kNbrs is a float as well.
        '''

        tspDraw.base.Annealer.__init__(self, nSteps, vertices, temperature, tempCool)

        self.kNbrs = kNbrs
        self.nbrsCool = nbrsCool

        # Make sure to train the Nearest Neighbors class.
        self._nearestNbrs = NearestNeighbors()
        self._nearestNbrs.fit(self.vertices.copy())

        # Conversion indices are originally just the identity function.
        self._origToCurrent = np.arange(self.nVertices)
        self._currentToOrig = np.arange(self.nVertices)

    def _updateState(self):
        tspDraw.base.Annealer._updateState(self)
        self.kNbrs *= self.nbrsCool

    def _makeRandomPair(self):
        '''
        Get a random pair of indices for vertices. The first index is chosen uniformly. The second
        index is chosen from the k-Nearest Neighbors of the first vertex. Note that we convert
        kNbrs to an Int to get the number of neighbors.

        Returns
        -------
        (Int, Int)
            The indices of the two random vertices. The first index will be less than the second.
        '''
        sameNum = True
        trivial = True
        kNbrs = int(self.kNbrs)

        # We loop until we have a choice that is two different indices and
        # doesn't include a trivial choice of the first and last indices.

        while sameNum or trivial:
        
            begin = np.random.randint(self.nVertices)
            beginV = self.vertices[begin].reshape(1,-1)

            # Find the neighbors of begin.
            _, nbrsI = self._nearestNbrs.kneighbors(beginV, n_neighbors = kNbrs)
            nbrsI = nbrsI.reshape(-1)

            # Randomly choose from the neighbors.

            end = np.random.randint(len(nbrsI))
            end = nbrsI[end]
            end = self._origToCurrent[end]

            # Check that our pair is acceptable.

            sameNum = (begin == end)
            trivial = (begin == 0) & (end == self.nVertices - 1)
            
        if begin < end:

            return begin, end

        else:

            return end, begin  

    def _makeMove(self, begin, end):
        '''
        Perform a reversal of the segment of the cycle between begin and end (inclusive). Also handles
        the effects on the conversions between the original and new indices (needed for nearest neighbor search).

        Parameters
        ----------
        begin : Int
            The index of the beginning of the segment.

        end : Int
            The index of the end of the segment.

        '''

        self.vertices[begin : end + 1] = np.flip(self.vertices[begin : end + 1], axis = 0)
        self._currentToOrig[begin : end + 1] = np.flip(self._currentToOrig[begin : end + 1], axis = 0)

        # Updating the conversion from original to current indices requires more than a flip.

        beforeFlip = self._currentToOrig[begin : end + 1]
        self._origToCurrent[beforeFlip] = np.arange(begin, end+1)

    def getInfoString(self):
        '''
        Get information on the current parameters of the annealing process as a string.

        Returns
        -------
        String
            Contains information on the energy, kNbrs, and the temperature. 
        '''

        info = tspDraw.base.Annealer.getInfoString(self)
        info += '\tkNbrs = ' + str(self.kNbrs)

        return info

