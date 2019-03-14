'''
Virtual base annealing class for basic simulated annealing behavior.
'''

import numpy as np

class Annealer:
    '''
    Virtual base class for doing simulated annealing. Handles temperature cooling,
    running an accepted move trial, finding energy differences, doing a warm restart that
    involves resetting counters.

    Sub-classes need to redefine the following methods:
        _make_random_pair()
        _make_move()

    Members
    -------
    n_steps : Int
        Total number of steps to use for one run (when iteration stops).

    vertices : Numpy array of shape (n_vertices, 2)
        The xy-coordinates of the vertices.

    temperature : Float
        The current temperature of the annealer. Used for computing probability of
        moving to a higher energy state.

    temp_cool : Float
        Multiplicatively changes the temperature (usually you want to reduce temperature) at
        every iteration.

    steps_processed : Int
        The number of steps processed in the run.

    n_vertices : Int
        The number of vertices in the cycle.
    '''
    def __init__(self, n_steps, vertices, temperature, temp_cool):

        # Members from Parameters

        self.n_steps = n_steps
        self.vertices = vertices
        self.temperature = temperature
        self.temp_cool = temp_cool

        self.steps_processed = 0
        self.n_vertices = len(vertices)

    def __iter__(self):
        return self

    def __next__(self):

        if self.steps_processed >= self.n_steps:
            raise StopIteration

        self._update_state()
        begin, end = self._make_random_pair()
        energy_diff = self._find_energy_difference(begin, end)
        if energy_diff < 0:
            proposal_accepted = True
        else:
            proposal_accepted = self._run_proposal_trial(energy_diff)

        if proposal_accepted:
            self._make_move(begin, end)

        return energy_diff

    def do_warm_restart(self):
        '''
        Reset the steps processed counter.
        '''
        self.steps_processed = 0

    def get_cycle(self):
        '''
        Get the vertices in the order they appear in the cycle.

        Returns
        -------
        Numpy array of shape (n_vertices, 2)
            The coordinates of the vertices for the order they appear in the cycle.
        '''
        cycle = self.vertices.copy()
        begin = self.vertices[0]
        cycle = np.concatenate([cycle, [begin]], axis = 0)
        return cycle

    def get_energy(self):
        '''
        Compute the energy (i.e. the length) of the current cycle.

        Returns
        -------
        Energy : Float
            The current energy.
        '''
        energy = np.linalg.norm(self.vertices[1:] - self.vertices[:-1], axis = 1).sum()
        energy += np.linalg.norm(self.vertices[0] - self.vertices[-1])
        return energy

    def get_info_string(self):
        '''
        Get a string for information of the current state of the iterator.

        Returns
        -------
        String
            Contains information for the energy and the temperature.
        '''

        energy = self.get_energy()
        info = 'Energy = ' + str(energy)
        info += '\tTemperature = ' + str(self.temperature)

        return info

    def _update_state(self):
        self.temperature *= self.temp_cool
        self.steps_processed += 1

    def _find_energy_difference(self, i, j):
        '''
        Find the energy (i.e. length) difference resulting from reversing the path between
        the ith vertex in the cycle with the jth vertex in the cycle. Both of the indices i and j
        should be greater than 0 (i.e. excluding the first vertex in the cycle) and less than
        self.n_vertices - 1 (i.e. excluding the last vertex in the cycle).

        Parameters
        ----------
        i : Int
            Index for the ith vertex in the cycle. Should be greater than 0 and less than
            self.n_vertices - 1.

        j : Int
            Index for the jth vertex in the cycle. Should be greater than 0 and less than
            self.n_vertices - 1.

        Returns
        -------
        Float
            The energy different (i.e. length difference) resulting from a proposed reversal.
        '''

        begin = self.vertices[i]
        if i > 0:
            begin_parent = self.vertices[i - 1]
        else:
            begin_parent = self.vertices[self.n_vertices - 1]

        end = self.vertices[j]
        if j < self.n_vertices - 1:
            end_child = self.vertices[j + 1]
        else:
            end_child = self.vertices[0]

        old_energy = np.linalg.norm(begin - begin_parent) + np.linalg.norm(end - end_child)
        new_energy = np.linalg.norm(begin - end_child) + np.linalg.norm(end - begin_parent)

        return new_energy - old_energy

    def _run_proposal_trial(self, energy_diff):
        '''
        Run the bernoulli trial for determining whether to accept a proposed reversal in the case
        that the reversal will result in an increase in energy.

        Parameters
        ----------
        energy_diff : Float
            The energy difference (i.e. lengthDifference) for the proposal. This should be greater
            than 0.

        Returns
        -------
        Bool
            Whether to accept the proposal based on the random bernoulli trial.
        '''

        prob = np.exp(-energy_diff / self.temperature)

        trial = np.random.uniform()

        return trial < prob

    def _make_random_pair(self):
        raise NotImplementedError()

    def _make_move(self, begin, end):
        raise NotImplementedError()
