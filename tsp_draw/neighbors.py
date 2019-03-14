'''
Annealer that selects a random vertex and then randomly selects a second vertex
from a number of the first's nearest neighbors.
'''
import numpy as np
from sklearn.neighbors import NearestNeighbors
import tsp_draw.base

##########################################
#### NeighborsAnnealer
##########################################

class Annealer(tsp_draw.base.Annealer):
    '''
    Modified simulated annealer that randomly selects a vertex and then randomly selects another
    vertex from the k-nearest neighbors of the first vertex. The point of this annealer is to do
    annealing when the current cycle is at a stage where the changes needed to be made are by
    switching vertices that are close ot each other.

    Members
    -------
    Members inherited from tsp_draw.base.Annealer

    k_nbrs : Float
        The number of neighbors to randomly select from. This is converted to an int when doing
        selection. It is a float because we do geometric cooling on the number of neighbors as we
        run each step.

    nbrs_cool : Float
        The factor to use to cool (decay) the number of neigbors. At each step, it is applied to
        k_nbrs via multiplication.

    _nearest_nbrs : class NearestNeighbors
        We use sci-kit-learn NearestNeighbors to find the nearest neighbors of vertices. This is
        trained on the original order of the vertices, so we need to deal with converting between
        the original order of the vertices to the current order of the vertices in the array.

    _orig_to_current : Numpy Array of Int of Shape (n_vertices)
        Array for converting from original indices to current indices in cycle. That is
        orig_to_current[i] is the current index of what was originally the ith vertex. This
        is needed for dealing with results of nearest neighbors search.

    _current_to_orig: Numpy Array of Int of Shape (n_vertices)
        Array for converting from the current index of a vertex to the original index of the vertex.
        That is current_to_orig[i] is the original index of what is now index i in the cycle. This
        is needed to update orig_to_current when doing a reversal.
    '''

    def __init__(self, nSteps, vertices, temperature, temp_cool, k_nbrs, nbrs_cool):
        '''
        Initializer. Make sure to train the nearest neighbors on the original order of the vertices.

        Parameters
        ----------

        nSteps : Int
            The total number of iterations to make.

        vertices : Numpy array of Floats of shape (n_vertices, 2)
            The vertices in their initial order.

        temperature : Float
            The initial temperature to use for the annealing.

        temp_cool : Float
            The cooling factor to apply to the temperature at each step; it is applied
            via multiplication. That is, we have geometric cooling.

        k_nbrs : Float
            The initial value for k_nbrs.

        nbrs_cool : Float
            The cooling factor (decay factor) for the number of neighbors; at each step it
            is applied to k_nbrs via multiplication. Note that k_nbrs is a float as well.
        '''

        tsp_draw.base.Annealer.__init__(self, nSteps, vertices, temperature, temp_cool)

        self.k_nbrs = k_nbrs
        self.nbrs_cool = nbrs_cool

        # Make sure to train the Nearest Neighbors class.
        self._nearest_nbrs = NearestNeighbors()
        self._nearest_nbrs.fit(self.vertices.copy())

        # Conversion indices are originally just the identity function.
        self._orig_to_current = np.arange(self.n_vertices)
        self._current_to_orig = np.arange(self.n_vertices)

    def _update_state(self):
        tsp_draw.base.Annealer._update_state(self)
        self.k_nbrs *= self.nbrs_cool

    def _make_random_pair(self):
        '''
        Get a random pair of indices for vertices. The first index is chosen uniformly. The second
        index is chosen from the k-Nearest Neighbors of the first vertex. Note that we convert
        k_nbrs to an Int to get the number of neighbors.

        Returns
        -------
        (Int, Int)
            The indices of the two random vertices. The first index will be less than the second.
        '''
        same_num = True
        trivial = True
        k_nbrs = int(self.k_nbrs)

        # We loop until we have a choice that is two different indices and
        # doesn't include a trivial choice of the first and last indices.

        while same_num or trivial:

            begin = np.random.randint(self.n_vertices)
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
        Perform a reversal of the segment of the cycle between begin and end (inclusive). Also
        handles the effects on the conversions between the original and new indices (needed for
        nearest neighbor search).

        Parameters
        ----------
        begin : Int
            The index of the beginning of the segment.

        end : Int
            The index of the end of the segment.

        '''

        self.vertices[begin : end + 1] = np.flip(self.vertices[begin : end + 1],
                                                 axis = 0)
        self._current_to_orig[begin : end + 1] = np.flip(self._current_to_orig[begin : end + 1],
                                                         axis = 0)

        # Updating the conversion from original to current indices requires more than a flip.

        before_flip = self._current_to_orig[begin : end + 1]
        self._orig_to_current[before_flip] = np.arange(begin, end+1)

    def get_info_string(self):
        '''
        Get information on the current parameters of the annealing process as a string.

        Returns
        -------
        String
            Contains information on the energy, k_nbrs, and the temperature.
        '''

        info = tsp_draw.base.Annealer.get_info_string(self)
        info += '\tk_nbrs = ' + Annealer._float_formatter.format(self.k_nbrs)

        return info
