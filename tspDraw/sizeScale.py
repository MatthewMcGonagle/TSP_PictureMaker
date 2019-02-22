import numpy as np
import tspDraw.base

def guessSettings(vertices, nStepsPerJob, nJobs = 10):
    '''
    Vertices should be pre-normalized.

    Estimates are based on assumption that vertices fill in a unit square area uniformly.

    Parameters
    ----------
    vertices : Numpy Array of Shape (nPoints, 2)
        Holds the xy-coordinates of the points on the path.

    nStepsPerJob : Int
        The number of steps to perform for each job.

    Returns
    -------
    settings : Dictionary of other parameters for Annealer().
    '''

    nVert = len(vertices) 
    expectedLength = np.sqrt(nVert)
    distances = np.linalg.norm(vertices[1:] - vertices[:-1], axis = -1)
    actualLength = distances.sum()
    segment = actualLength / nVert

    differenceLength = np.abs(expectedLength - actualLength)
    proportionToMakeHalfChance = 0.001
    temperature = proportionToMakeHalfChance * actualLength / np.log(2)
    cooling = np.abs(np.log(actualLength) - np.log(expectedLength))
    cooling = np.exp(-cooling / nStepsPerJob / nJobs)
    # temperature = proportionToMakeHalfChance * differenceLength / np.log(2) 
    temperature = 3 * segment / np.log(2)
    cooling = np.exp(np.log(1.0/3) / nStepsPerJob / nJobs)
   
    # newProportion = proportionToMakeHalfChance * 0.99
    # cooling = np.exp(np.log(newProportion / proportionToMakeHalfChance) / nStepsPerJob) 
  
    initScale = np.percentile(distances, 99.8)
    finalScale = np.percentile(distances, 99.5) 
    finalScale = segment / 2 
    print(nVert)
    print(initScale, finalScale)
    sizeCooling = np.exp(np.log(finalScale / initScale) /nStepsPerJob / nJobs)

    settings = {'temperature' : temperature,
                'tempCool' : cooling,
                'sizeScale' : initScale,
                'sizeCool' : sizeCooling
               } 

    return settings

class Annealer(tspDraw.base.Annealer):
    '''
    An iterator for performing annealing based on a size scale. The annealing is done on a pool of vertices
    that are on an edge of the cycle that is at least as long as the current size scale; so one should think
    of the annealing as starting with those edges of the cycle that are large.

    Note the vertex pool is NOT updated each step of the iteration; this would be too costly. Instead the 
    vertex poolis updated upon a warm restart. 
    
    Members
    -------
    Members Inherited from tspDraw.base.Annealer
 
    sizeScale : Float
        The current size scale. Used when setting up the pool of vertices between jobs.
        At the time of construction, the pool consists of vertices touching a segment
        of size at least as large as sizeScale.
    
    sizeCool : Float
        The multiplicative factor to use to lower the sizeScale at each step. 
    
    poolV : Numpy array of Int of shape (nPool)
        The indices of the elements of vOrder that appear in the pool.
    
    nPool : Int
        The number of vertices in the pool.
    
    '''

    def __init__(self, nSteps, vertices, temperature, tempCool, sizeScale, sizeCool):
        '''
        Set up the total number of steps that the iterator will take as well as the cooling. 

        Parameters
        ----------
        nSteps : Int
            The total number of steps the iterator will take.
        
        vertices : Numpy array of shape (nVertices, 2)
            The xy-coordinates of the vertices.

        temperature : Float
            The initial temperature of the anealing process.

        tempCool : Float
            The cooling parameter to apply to the temperature at each step. The cooling is applied via multiplication.

        sizeScale : Float
            The size scale that the annealer will start at. Starts by looking at vertices that are connected to an edge 
            of size atleast as large as the size scale.

        sizeCool : Float
            The rate (or decay) of the size scale. The size cooling is applied via multiplication by sizeCool.

        '''
        tspDraw.base.Annealer.__init__(self, nSteps, vertices, temperature, tempCool)  
        self.sizeScale = sizeScale
        self.sizeCool = sizeCool

        # Initialize the pool of vertices for the size scale to be None. Then set up the scale pool 
        # based on the initial scale size.
        self.poolV = None
        self.nPool = 0 
        self._findScalePool()

    def doWarmRestart(self):
        '''
        Do a warm restart of the iterator. This also updates the size scale vertex pool.
        '''
        tspDraw.base.Annealer.doWarmRestart(self)
        self._findScalePool()

    def _updateState(self):
        tspDraw.base.Annealer._updateState(self)
        self.sizeScale *= self.sizeCool

    def _findScalePool(self):
        '''
        Reset the pool of vertices for annealing based on the current cycle edge sizes and
        the current size scale.

        If the scale pool has only one vertex then a ValueError exception is raised.
        '''

        # First find the interior differences, i.e. the vector differences between vertices in the
        # cycle excluding the difference between the final and last vertex.
        interiorDiff = self.vertices[1:] - self.vertices[:-1] 

        # Next find the vector difference between the first vertex and the last vertex. 
        joinDiff = self.vertices[-1] - self.vertices[0]

        # Find the forward differences and backward differences for each vertex.
        forwardDiff = np.concatenate([interiorDiff, [joinDiff]], axis = 0) 
        backwardDiff = np.concatenate([[joinDiff], interiorDiff], axis = 0)

        # Find the forward lengths and backward lengths.
        forwardDist = np.linalg.norm(forwardDiff, axis = -1)
        backwardDist = np.linalg.norm(backwardDiff, axis = -1)

        # Find which vertices are in the pool based on whether the forward
        # length or backward length is large enough.

        verticesInPool = (forwardDist > self.sizeScale) | (backwardDist > self.sizeScale)

        self.poolV = np.arange(self.nVertices)[verticesInPool]
        self.nPool = len(self.poolV)

        if self.nPool < 2:

            raise ValueError('Size scale pool has less than two vertices.')

    def _makeRandomPair(self):
        ''' 
        Get a random pair of vertices from the pool of vertices in the current size pool. This
        should only be called if there are atleast two vertices in the size scale pool.

        Returns
        -------
        (Int, Int)
            A pair of indices for which vertices to use from the pool. The indices are for
            the array self.poolV, not the original vertices array. Also the first index
            is always less than the second.
        '''

        sameNum = True

        while sameNum:

           begin = np.random.randint(self.nPool)
           end = np.random.randint(self.nPool) 

           begin = self.poolV[begin]
           end = self.poolV[end]

           sameNum = (begin == end) 

        if begin < end: 

            return begin, end

        else:

            return end, begin

    def _makeMove(self, i, j):
        '''
        Reverse the order of vertices between the ith vertex in the ith vertex in the cycle and the jth
        vertex in the cycle. Not that this reverses elements contained in the array self.vOrder and
        not the original array of vertices.

        Also note that we don't have to do anything to the pool of vertices. Some elements of the pool
        will point to different vertices, but this is really just a rearrangement of the vertices 
        contained in the pool. This will have no effect on random selection which is the only
        purpose of the pool. 

        Parameters
        ----------
        i : Int
            The index of the element in self.vOrder for the beginning of the reversal segment; i.e. the ith vertex
            in the cycle. The index i should be less than the index j.

        j : Int
            The index of the elemet in self.vOrder for the end of the reversal segment; i.e. the jth vertex in the
            cycle. The index j should be greater than the index i. 
        '''

        # Note that flip two vertices in the vertex pool keeps them in the pool; Note that we do not require
        # that self.poolV is ordered.
        self.vertices[i : j + 1] = np.flip(self.vertices[i : j + 1], axis = 0)

    def getInfoString(self):
        '''
        Get a string for information of the current state of the iterator.

        Returns
        -------
        String
            Contains information for the energy, number of vertices in the size scale pool, and the 
            temperature.
        '''

        info = tspDraw.base.Annealer.getInfoString(self)
        info += '\tsizeScale = ' + str(self.sizeScale)
        info += '\tnPool = ' + str(self.nPool)
        return info

