'''
Annealer that uses a candidate pool of vertices that is based on a certain size scale,
then selects a random neighbor of random vertex from the candidate pool.
'''
import numpy as np
from sklearn.neighbors import NearestNeighbors
import tspDraw.size_scale

class Annealer(tspDraw.size_scale.Annealer):
    '''
    Annealer that uses a candidate pool of vertices that is based on a certain size scale,
    then selects a random neighbor of random vertex from the candidate pool.
    '''

    def __init__(self, nSteps, vertices, temperature, temp_cool, size_scale,
                 sizeCool, k_nbrs, nbrs_cool):

        tspDraw.size_scale.Annealer.__init__(self, nSteps, vertices, temperature,
                                             temp_cool, size_scale, sizeCool)
        self.k_nbrs = k_nbrs
        self.nbrs_cool = nbrs_cool

        self._nearest_nbrs = NearestNeighbors()
        self._nearest_nbrs.fit(vertices.copy())

        self._orig_to_current = np.arange(self.n_vertices)
        self._current_to_orig = np.arange(self.n_vertices)
        self._pool_replace = None

    def _update_state(self):
        '''
        Cool the neighbors number and update the state inherited from
        tspDraw.size_scale.Annealer.
        '''
        tspDraw.size_scale.Annealer._update_state(self)
        self.k_nbrs *= self.nbrs_cool

    def _make_random_pair(self):
        '''
        First we select a random vertex from the pool of candidate
        vertices. Then we select a random vertex from a number of that
        vertex's nearest neighbors.

        Returns
        -------
        (begin, end) : Pair of Int
            Indices in the current cycle of the two random vertices. Guaranteed that
            begin < end.
        '''
        same_num = True
        trivial = True
        k_nbrs = int(self.k_nbrs)

        # We loop until we have a choice that is two different indices and
        # doesn't include a trivial choice of the first and last indices.

        while same_num or trivial:

            begin = np.random.randint(self.n_pool)
            self._pool_replace = begin
            begin = self.pool_v[begin]
            begin_v = self.vertices[begin].reshape(1, -1)

            # Find the neighbors of begin.
            _, nbrs_i = self._nearest_nbrs.kneighbors(begin_v, n_neighbors = k_nbrs)
            nbrs_i = nbrs_i.reshape(-1)

            # Randomly choose from the neighbors.

            end = np.random.randint(len(nbrs_i))
            end = nbrs_i[end]
            end = self._orig_to_current[end]

            # Check that our pair is acceptable.

            same_num = (begin == end)
            trivial = (begin == 0) & (end == self.n_vertices - 1)

        if begin < end:
            pair = (begin, end)

        else:
            pair = (end, begin)

        return pair

    def _make_move(self, begin, end):
        '''
        Perform a reversal of the segment of the cycle between begin and end (inclusive).
        Also handles the effects on the conversions between the original and new indices
        (needed for nearest neighbor search).

        Parameters
        ----------
        begin : Int
            The index of the beginning of the segment.

        end : Int
            The index of the end of the segment.
        '''

        self.vertices[begin : end + 1] = np.flip(self.vertices[begin : end + 1], axis = 0)
        self._current_to_orig[begin : end + 1] = np.flip(self._current_to_orig[begin : end + 1],
                                                         axis = 0)

        # Updating the conversion from original to current indices requires more than a flip.

        before_flip = self._current_to_orig[begin : end + 1]
        self._orig_to_current[before_flip] = np.arange(begin, end+1)

        self.pool_v[self._pool_replace] = end

    def get_info_string(self):
        '''
        Get information on the current parameters of the annealing process as a string.

        Returns
        -------
        String
            Contains information on the energy, k_nbrs, and the temperature.
        '''

        info = tspDraw.size_scale.Annealer.get_info_string(self)
        info += '\tk_nbrs = ' + str(self.k_nbrs)

        return info
