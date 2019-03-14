'''
Annealer for only dealing with vertices that are adjacent to edges that are large
enough determined by a specified size scale. The idea is that this allows the
annealer to concentrate on dealing with only the largest of segments in the
cycle.

Also has functions for guessing correct initial settings of the annealer.
'''

import numpy as np
import tsp_draw.base
import tsp_draw.exception

def _guess_temperature_settings(n_jobs, n_steps_per_job, segment_length):
    '''
    Guess the initial temperature and temperature cooling based on the
    current average segment length.

    Parameters
    ----------
    n_jobs : Int
        The number of jobs to do the cooling over.

    n_steps_per_job : Int
        The number of steps to cool over for each job.

    segment_length : Float
        The current average segment length between consecutive vertices
        in the cycle.

    Returns
    -------
    (temperature, temp_cooling) : (Float, Float)
        The temperature and temperature cooling settings.
    '''
    # Old method that tries to guess cooling based on actual length vs expected length.
    # expected_length = np.sqrt(n_vert)
    # difference_length = np.abs(expected_length - actual_length)
    # proportion_to_make_half_chance = 0.001
    # temperature = proportion_to_make_half_chance * actual_length / np.log(2)
    # cooling = np.abs(np.log(actual_length) - np.log(expected_length))
    # cooling = np.exp(-cooling / n_steps_per_job / n_jobs)
    cooling = np.exp(np.log(1.0/3) / n_steps_per_job / n_jobs)
    temperature = 3 * segment_length / np.log(2)
    return temperature, cooling

def _guess_size_settings(n_jobs, n_steps_per_job, distances, segment_length):
    '''
    Guess the size scale and size cooling based on the current distances and
    the average segment length.

    Parameters
    ----------
    n_jobs : Int
        The number of jobs to do the cooling over.

    n_steps_per_job : Int
        The number of steps per job to do cooling over.

    distances : Numpy array of Float
        The distances of each segment in the cycle.

    segment_length : Float
        The average segment length between consecutive vertices in the cycle.

    Returns
    -------
    (size_scale, size_cool) : (Float, Float)
        The size scale and the size scale cooling settings.
    '''
    init_scale = np.percentile(distances, 99.8)
    final_scale = segment_length / 2
    print(init_scale, final_scale)
    size_cooling = np.exp(np.log(final_scale / init_scale) / n_steps_per_job / n_jobs)
    return init_scale, size_cooling

def guess_settings(vertices, n_steps_per_job, n_jobs = 10):
    '''
    Vertices should be pre-normalized.

    Estimates are based on assumption that vertices fill in a unit square area uniformly.

    Parameters
    ----------
    vertices : Numpy Array of Shape (nPoints, 2)
        Holds the xy-coordinates of the points on the path.

    n_steps_per_job : Int
        The number of steps to perform for each job.

    n_jobs : Int
        The number of jobs to do.

    Returns
    -------
    settings : Dictionary of other parameters for Annealer().
    '''

    n_vert = len(vertices)
    distances = np.linalg.norm(vertices[1:] - vertices[:-1], axis = -1)
    actual_length = distances.sum()
    segment_length = actual_length / n_vert

    temperature, temp_cool = _guess_temperature_settings(n_jobs, n_steps_per_job, segment_length)
    size_scale, size_cool = _guess_size_settings(n_jobs, n_steps_per_job, distances, segment_length)

    settings = {'temperature' : temperature,
                'temp_cool' : temp_cool,
                'size_scale' : size_scale,
                'size_cool' : size_cool
               }

    return settings

class Annealer(tsp_draw.base.Annealer):
    '''
    An iterator for performing annealing based on a size scale. The annealing is done on a pool of
    vertices that are on an edge of the cycle that is at least as long as the current size scale;
    so one should think of the annealing as starting with those edges of the cycle that are large.

    Note the vertex pool is NOT updated each step of the iteration; this would be too costly.
    Instead the vertex pool is updated upon a warm restart.

    Members
    -------
    Members Inherited from tsp_draw.base.Annealer

    size_scale : Float
        The current size scale. Used when setting up the pool of vertices between jobs.
        At the time of construction, the pool consists of vertices touching a segment
        of size at least as large as size_scale.

    size_cool : Float
        The multiplicative factor to use to lower the size_scale at each step.

    pool_v : Numpy array of Int of shape (n_pool)
        The indices of the elements of vOrder that appear in the pool.

    n_pool : Int
        The number of vertices in the pool.
    '''

    def __init__(self, n_steps, vertices, temperature, temp_cool, size_scale, size_cool):
        '''
        Set up the total number of steps that the iterator will take as well as the cooling.

        Parameters
        ----------
        n_steps : Int
            The total number of steps the iterator will take.

        vertices : Numpy array of shape (nVertices, 2)
            The xy-coordinates of the vertices.

        temperature : Float
            The initial temperature of the anealing process.

        temp_cool : Float
            The cooling parameter to apply to the temperature at each step. The cooling is applied
            via multiplication.

        size_scale : Float
            The size scale that the annealer will start at. Starts by looking at vertices that are
            connected to an edge of size atleast as large as the size scale.

        size_cool : Float
            The rate (or decay) of the size scale. The size cooling is applied via multiplication
            by size_cool.
        '''
        tsp_draw.base.Annealer.__init__(self, n_steps, vertices, temperature, temp_cool)
        self.size_scale = size_scale
        self.size_cool = size_cool

        # Initialize the pool of vertices for the size scale to be None. Then set up the scale pool
        # based on the initial scale size.
        self.pool_v = None
        self.n_pool = 0
        self._find_scale_pool()

    def do_warm_restart(self):
        '''
        Do a warm restart of the iterator. This also updates the size scale vertex pool.
        '''
        tsp_draw.base.Annealer.do_warm_restart(self)
        self._find_scale_pool()

    def _update_state(self):
        tsp_draw.base.Annealer._update_state(self)
        self.size_scale *= self.size_cool

    def _find_scale_pool(self):
        '''
        Reset the pool of vertices for annealing based on the current cycle edge sizes and
        the current size scale.

        If the scale pool has only one vertex then a ValueError exception is raised.
        '''

        # First find the interior differences, i.e. the vector differences between vertices in the
        # cycle excluding the difference between the final and last vertex.
        interior_diff = self.vertices[1:] - self.vertices[:-1]

        # Next find the vector difference between the first vertex and the last vertex.
        join_diff = self.vertices[-1] - self.vertices[0]

        # Find the forward differences and backward differences for each vertex.
        forward_diff = np.concatenate([interior_diff, [join_diff]], axis = 0)
        backward_diff = np.concatenate([[join_diff], interior_diff], axis = 0)

        # Find the forward lengths and backward lengths.
        forward_dist = np.linalg.norm(forward_diff, axis = -1)
        backward_dist = np.linalg.norm(backward_diff, axis = -1)

        # Find which vertices are in the pool based on whether the forward
        # length or backward length is large enough.

        vertices_in_pool = (forward_dist > self.size_scale) | (backward_dist > self.size_scale)

        self.pool_v = np.arange(self.n_vertices)[vertices_in_pool]
        self.n_pool = len(self.pool_v)

        if self.n_pool < 2:

            raise tsp_draw.exception.VertexPoolTooSmall(self.n_pool,
                                                       "Vertex pool is too small.") 

    def _make_random_pair(self):
        '''
        Get a random pair of vertices from the pool of vertices in the current size pool. This
        should only be called if there are atleast two vertices in the size scale pool.

        Returns
        -------
        (Int, Int)
            A pair of indices for which vertices to use from the pool. The indices are for
            the array self.pool_v, not the original vertices array. Also the first index
            is always less than the second.
        '''

        same_num = True

        while same_num:
            begin = np.random.randint(self.n_pool)
            end = np.random.randint(self.n_pool)

            begin = self.pool_v[begin]
            end = self.pool_v[end]

            same_num = (begin == end)

        if begin < end:
            pair = begin, end

        else:
            pair = end, begin

        return pair

    def _make_move(self, begin, end):
        '''
        Reverse the order of vertices between the vertex begin in the cycle and the vertex end in
        the cycle (begin < end). Not that this reverses elements contained in the array
        self.vOrder and not the original array of vertices.

        Also note that we don't have to do anything to the pool of vertices. Some elements of the
        pool will point to different vertices, but this is really just a rearrangement of the
        vertices contained in the pool. This will have no effect on random selection which is the
        only purpose of the pool.

        Parameters
        ----------
        begin : Int
            The index of the element in self.vOrder for the beginning of the reversal segment; The
            index begin should be less than the index end.

        end : Int
            The index of the element in self.vOrder for the end of the reversal segment; The index
            end should be greater than the index begin.
        '''

        # Note that flip two vertices in the vertex pool keeps them in the pool; Note that we do not
        # require that self.pool_v is ordered.
        self.vertices[begin : end + 1] = np.flip(self.vertices[begin : end + 1], axis = 0)

    def get_info_string(self):
        '''
        Get a string for information of the current state of the iterator.

        Returns
        -------
        String
            Contains information for the energy, number of vertices in the size scale pool, and the
            temperature.
        '''

        info = tsp_draw.base.Annealer.get_info_string(self)
        info += '\tsize_scale = ' + str(self.size_scale)
        info += '\tn_pool = ' + str(self.n_pool)
        return info
