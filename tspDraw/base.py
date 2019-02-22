import numpy as np

class Annealer:

    def __init__(self, nSteps, vertices, temperature, tempCool):

        # Members from Parameters

        self.nSteps = nSteps
        self.vertices = vertices
        self.temperature = temperature
        self.tempCool = tempCool

        self.stepsProcessed = 0
        self.nVertices = len(vertices)
        

    def __iter__(self):
        return self

    def __next__(self):

        if self.stepsProcessed >= self.nSteps:
            raise StopIteration

        self._updateState()
        begin, end = self._makeRandomPair()
        energyDiff = self._findEnergyDifference(begin, end)
        if energyDiff < 0:
            proposalAccepted = True
        else:
            proposalAccepted = self._runProposalTrial(energyDiff)
        
        if proposalAccepted:
            self._makeMove(begin, end)

        return energyDiff

    def doWarmRestart(self):
        self.stepsProcessed = 0

    def getCycle(self):
        '''
        Get the vertices in the order they appear in the cycle. 

        Returns
        -------
        Numpy array of shape (nVertices, 2)
            The coordinates of the vertices for the order they appear in the cycle. 
        '''
        cycle = self.vertices.copy()
        begin = self.vertices[0]
        cycle = np.concatenate([cycle, [begin]], axis = 0)
        return cycle

    def getEnergy(self):
        energy = np.linalg.norm(self.vertices[1:] - self.vertices[:-1], axis = 1).sum()
        energy += np.linalg.norm(self.vertices[0] - self.vertices[-1])
        return energy

    def getInfoString(self):
        '''
        Get a string for information of the current state of the iterator.

        Returns
        -------
        String
            Contains information for the energy and the temperature.
        '''

        energy = self.getEnergy()
        info = 'Energy = ' + str(energy)
        info += '\tTemperature = ' + str(self.temperature)

        return info

    def _updateState(self):
        self.temperature *= self.tempCool
        self.stepsProcessed += 1

    def _findEnergyDifference(self, i, j):
        '''
        Find the energy (i.e. length) difference resulting from reversing the path between
        the ith vertex in the cycle with the jth vertex in the cycle. Both of the indices i and j 
        should be greater than 0 (i.e. excluding the first vertex in the cycle) and less than
        self.nVertices - 1 (i.e. excluding the last vertex in the cycle).

        Parameters
        ----------
        i : Int
            Index for the ith vertex in the cycle. Should be greater than 0 and less than self.nVertices - 1.

        j : Int
            Index for the jth vertex in the cycle. Should be greater than 0 and less than self.nVertices - 1.

        Returns
        -------
        Float
            The energy different (i.e. length difference) resulting from a proposed reversal.
        '''

        begin = self.vertices[i]
        if i > 0:
            beginParent = self.vertices[i - 1]
        else:
            beginParent = self.vertices[self.nVertices - 1]

        end = self.vertices[j]
        if j < self.nVertices - 1:
            endChild = self.vertices[j + 1]
        else:
            endChild = self.vertices[0]
       
        oldEnergy = np.linalg.norm(begin - beginParent) + np.linalg.norm(end - endChild)
        newEnergy = np.linalg.norm(begin - endChild) + np.linalg.norm(end - beginParent)         

        energyDiff = newEnergy - oldEnergy

        return energyDiff

    def _runProposalTrial(self, energyDiff):
        '''
        Run the bernoulli trial for determining whether to accept a proposed reversal in the case
        that the reversal will result in an increase in energy.

        Parameters
        ----------
        energyDiff : Float
            The energy difference (i.e. lengthDifference) for the proposal. This should be greater than 0.

        Returns
        -------
        Bool
            Whether to accept the proposal based on the random bernoulli trial.
        '''

        prob = np.exp(-energyDiff / self.temperature)

        trial = np.random.uniform()

        return (trial < prob)

    def _makeRandomPair(self):
        raise Exception("_getRandomPair() is still virtual")

    def _makeMove(self, begin, end):
        raise Exception("_makeMove() is still virtual") 
