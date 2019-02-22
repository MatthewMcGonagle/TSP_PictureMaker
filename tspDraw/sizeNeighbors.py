import numpy as np
from sklearn.neighbors import NearestNeighbors
import tspDraw.sizeScale

class Annealer( tspDraw.sizeScale.Annealer ):

    def __init__(self, nSteps, vertices, temperature, tempCool, sizeScale, sizeCool, kNbrs, nbrsCool):
         
        tspDraw.sizeScale.Annealer.__init__(self, nSteps, vertices, temperature, tempCool, sizeScale, sizeCool)
        self.kNbrs = kNbrs
        self.nbrsCool = nbrsCool
 
        self._nearestNbrs= NearestNeighbors() 
        self._nearestNbrs.fit(vertices.copy())
        
        self._origToCurrent = np.arange(self.nVertices)
        self._currentToOrig = np.arange(self.nVertices)

    def _updateState(self):
        tspDraw.sizeScale.Annealer._updateState(self)
        self.kNbrs *= self.nbrsCool

    def _makeRandomPair(self):
        sameNum = True
        trivial = True
        kNbrs = int(self.kNbrs)

        # We loop until we have a choice that is two different indices and
        # doesn't include a trivial choice of the first and last indices.

        while sameNum or trivial:
        
            begin = np.random.randint(self.nPool)
            self._poolReplace = begin
            begin = self.poolV[begin]
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

        self.poolV[self._poolReplace] = end

    def getInfoString(self):
        '''
        Get information on the current parameters of the annealing process as a string.

        Returns
        -------
        String
            Contains information on the energy, kNbrs, and the temperature. 
        '''

        info = tspDraw.sizeScale.Annealer.getInfoString(self)
        info += '\tkNbrs = ' + str(self.kNbrs)

        return info

