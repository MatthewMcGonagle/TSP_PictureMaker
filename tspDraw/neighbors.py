import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

##########################################
#### NeighborsAnnealer
##########################################

class NeighborsAnnealer:
    '''
    Modified simulated annealer that randomly selects a vertex and then randomly selects another vertex from the
    k-nearest neighbors of the first vertex. The point of this annealer is to do annealing when the current
    cycle is at a stage where the changes needed to be made are by switching vertices that are close ot each other.    

    Members
    -------
    nSteps : Int
        The total number of steps to make.
    
    nProcessed : Int
        The total number of steps made so far.

    vertices : Numpy array of Floats of Shape (nVertices, 2)
        The vertices in their original order.

    nVertices : Int
        The number of vertices.

    temperature : Float
        The temperature used for annealing.

    cooling : Float
        The cooling factor for the temperature. At each step, it is applied to the temperature via multiplication.    
        That is, we use geometric cooling.
   
    kNbrs : Float 
        The number of neighbors to randomly select from. This is converted to an int when doing selection. It is
        a float because we do geometric cooling on the number of neighbors as we run each step. 

    nbrsCooling : Float
        The factor to use to cool (decay) the number of neigbors. At each step, it is applied to kNbrs via 
        multiplication.        

    nearestNbrs : class NearestNeighbors
        We use sci-kit-learn NearestNeighbors to find the nearest neighbors of vertices. This is trained on the
        original order of the vertices, so we need to deal with converting between the original order
        of the vertices to the current order of the vertices in the array.

    origToCurrent : Numpy Array of Int of Shape (nVertices)
        Array for converting from original indices to current indices in cycle. That is
        origToCurrent[i] is the current index of what was originally the ith vertex. This
        is needed for dealing with results of nearest neighbors search. 

    currentToOrig: Numpy Array of Int of Shape (nVertices)
        Array for converting from the current index of a vertex to the original index of the vertex.
        That is currentToOrig[i] is the original index of what is now index i in the cycle. This
        is needed to update origToCurrent when doing a reversal. 
    '''

    def __init__(self, nSteps, vertices, temperature, cooling, kNbrs, nbrsCooling):
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

        cooling : Float
            The cooling factor to apply to the temperature at each step; it is applied
            via multiplication. That is, we have geometric cooling. 

        kNbrs : Float
            The initial value for kNbrs.

        nbrsCooling : Float
            The cooling factor (decay factor) for the number of neighbors; at each step it
            is applied to kNbrs via multiplication. Note that kNbrs is a float as well.
        '''

        self.nSteps = nSteps
        self.nProcessed = 0

        self.vertices = vertices
        self.nVertices = len(vertices)

        self.temperature = temperature
        self.cooling = cooling
        self.kNbrs = kNbrs
        self.nbrsCooling = nbrsCooling

        # Make sure to train the Nearest Neighbors class.
        self.nearestNbrs = NearestNeighbors()
        self.nearestNbrs.fit(self.vertices.copy())
        self.origToCurrent = np.arange(self.nVertices)
        self.currentToOrig = np.arange(self.nVertices)

    def __iter__(self):
        '''
        Get an iterator reference

        Returns
        -------
        self
        '''
        return self

    def __next__(self):
        '''
        Do the next step of annealing. For the annealing we randomly pick a vertex and then randomly pick from the kNeighbors
        of that vertex. Note that we convert kNbrs to an Int to determing the number of neighbors.
        '''

        if self.nProcessed >= self.nSteps:

            raise StopIteration

        # Do annealing parameter updates.

        self.nProcessed += 1
        self.temperature *= self.cooling
        self.kNbrs *= self.nbrsCooling 

        # Get a random pair of vertices and find the energy difference if we were
        # to reverse the part of the cycle between them.
        begin, end = self.__getRandomPair()  
        energyDiff = self.__getEnergyDifference(begin, end)

        # Determine whether to accept the proposed reversal based on the energy difference.

        proposalAccepted = False

        if energyDiff < 0:
            proposalAccepted = True

        else:

            proposalAccepted = self.__runProposalTrial(energyDiff) 

        if proposalAccepted:

            self.__reverse(begin, end)
        

    def __runProposalTrial(self, energyDiff):
        '''
        Run the bernoulli trial to determine whether we accept a proposed reversal in the case
        that this reversal results in an increase in the length of the cycle.

        Parameters
        ----------
        energy Diff : Float
            The difference in cycle length if we accept the proposed reversal. This should be greater than 0.

        Returns
        -------
        Bool
            Whether the proposed reversal was accepted.
        '''

        trial = np.random.uniform()
        probAccept = np.exp(-energyDiff / self.temperature)
        if trial < probAccept:
            return True
        else:
            return False
        
    def __getEnergyDifference(self, begin, end):
        '''
        Find the energy difference (length difference) if we reverse the order of the vertices between begin
        and end (inclusive).

        Parameters
        ----------
        begin : Int
            The index of the starting vertex. Should be less than end.

        end : Int
            The index of the ending vertex. Should be greater than begin.

        Returns
        -------
        Float
            The energy difference (length difference).
        '''
 
        # If the begin is vertex 0, then we need to handle wrapping around to the end. 

        if begin > 0: 
            origDiffbegin = self.vertices[begin] - self.vertices[begin - 1] 
            newDiffbegin = self.vertices[end] - self.vertices[begin - 1]
        else:
            origDiffbegin = self.vertices[begin] - self.vertices[-1]
            newDiffbegin = self.vertices[end] - self.vertices[-1]

        # If end is the last vertex, then we need to handle wrapping around to the beginning.

        if end < self.nVertices - 1:
            origDiffend = self.vertices[end] - self.vertices[end + 1]
            newDiffend = self.vertices[begin] - self.vertices[end + 1]
        else:
            origDiffend = self.vertices[end] - self.vertices[0]
            newDiffend = self.vertices[begin] - self.vertices[0]

        origEnergy = np.linalg.norm(origDiffbegin) + np.linalg.norm(origDiffend)
        newEnergy = np.linalg.norm(newDiffbegin) + np.linalg.norm(newDiffend)
        energyDiff = newEnergy - origEnergy
        
        return energyDiff

    def __getRandomPair(self):
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
            _, nbrsI = self.nearestNbrs.kneighbors(beginV, n_neighbors = kNbrs)
            nbrsI = nbrsI.reshape(-1)

            # Randomly choose from the neighbors.

            end = np.random.randint(len(nbrsI))
            end = nbrsI[end]
            end = self.origToCurrent[end]

            # Check that our pair is acceptable.

            sameNum = (begin == end)
            trivial = (begin == 0) & (end == self.nVertices - 1)
            
        if begin < end:

            return begin, end

        else:

            return end, begin  

    def __reverse(self, begin, end):
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
        self.currentToOrig[begin : end + 1] = np.flip(self.currentToOrig[begin : end + 1], axis = 0)

        # Updating the conversion from original to current indices requires more than a flip.

        origIndices = self.currentToOrig[begin : end + 1]
        self.origToCurrent[origIndices] = np.arange(begin, end+1)

    def doWarmRestart(self):
        '''
        Do a warm restart. We only need to update the number of steps processed.
        '''

        self.nProcessed = 0 

    def getCycle(self):
        '''
        Return the cycle. Note that the cycle is closed so that the first vertex will match the last vertex.

        Returns
        -------
        Number Array of Floats of Shape (nVertices + 1, 2)
            The xy-coordinates of the vertices as they appear in the order of the cycle. Note that first
            vertex appears again at the end of the array to make the cycle closed.
        '''

        cycle = self.vertices.copy()
        cycle = np.concatenate([cycle, [self.vertices[0]] ], axis = 0)

        return cycle 

    def getEnergy(self):
        '''
        Get the energy (i.e. length) of the current cycle)
        
        Returns
        -------
        Float
            The length of the current cycle.
        '''

        differences = self.vertices[1:] - self.vertices[:-1]
        energy = np.linalg.norm(differences, axis = -1).sum()
        energy += np.linalg.norm(self.vertices[0] - self.vertices[-1])

        return energy

    def getInfoString(self):
        '''
        Get information on the current parameters of the annealing process as a string.

        Returns
        -------
        String
            Contains information on the energy, kNbrs, and the temperature. 
        '''

        info = 'Energy = ' + str(self.getEnergy())
        info += '\tkNbrs = ' + str(self.kNbrs)
        info += '\tTemperature = ' + str(self.temperature)

        return info

