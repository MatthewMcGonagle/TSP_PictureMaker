import unittest
import numpy as np
import sys

import base
sys.path.append('..')
import tsp_draw.size_scale
import fake_random

class TestAnnealerMethods(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        angles = np.linspace(0, 2 * np.pi, 8)[:-1]
        self.vertices = [[np.cos(3 * angle), np.sin(3 * angle)] for angle in angles]
        self.vertices = np.array(self.vertices)
        self.params = {'n_steps' : 3, 'vertices' : self.vertices, 'temperature' : 0.001,
                       'temp_cool' : 0.99, 'size_scale' : 0.1, 'size_cool' : 0.99,
                       'rand_state' : fake_random.State([])}

    def test_update_state(self):
        annealer = tsp_draw.size_scale.Annealer(**self.params)
        annealer._update_state()
        self.assertEqual(annealer.temperature,
                         self.params['temperature'] * self.params['temp_cool'])
        self.assertEqual(annealer.size_scale,
                         self.params['size_scale'] * self.params['size_cool'])

    def test_find_scale_pool(self):
        vertices = np.array([[0,0], [1,0], [3,0], [4,0], [5,0], [5,3], [4,3],
                             [3,3], [2,3], [1,3], [0,3]])
        params = self.params.copy()
        params['vertices'] = vertices
        params['size_scale'] = 1.0
        annealer = tsp_draw.size_scale.Annealer(**params)

        # Test that the scale pool is correct.
        annealer._find_scale_pool()
        true_pool_v = np.array([0, 1, 2, 4, 5, 10])
        np.testing.assert_equal(annealer.pool_v, true_pool_v)
        self.assertEqual(annealer.n_pool, len(true_pool_v))

        # Test that there is an exception for too little vertices.
        with self.assertRaises(tsp_draw.exception.VertexPoolTooSmall):
            annealer.size_scale = 3.0
            annealer._find_scale_pool()

    def test_make_random_pair(self):
        vertices = np.array([[0,0], [1,0], [3,0], [4,0], [5,0], [5,3], [4,3],
                             [3,3], [2,3], [1,3], [0,3]])
        params = self.params.copy()
        params['vertices'] = vertices
        params['size_scale'] = 1.0
        int_stack = [0, 0, 1, 1, 5, 3][::-1]
        params['rand_state'] = fake_random.State(int_stack = int_stack)
        annealer = tsp_draw.size_scale.Annealer(**params)
        true_pool_v = np.array([0, 1, 2, 4, 5, 10])

        true_pair = (4, 10)
        annealer._find_scale_pool()
        pair = annealer._make_random_pair()
        self.assertEqual(true_pair, pair)

    def test_make_move(self):
        vertices = np.array([[0,0], [1,0], [3,0], [4,0], [5,0], [5,3], [4,3],
                             [3,3], [2,3], [1,3], [0,3]])
        params = self.params.copy()
        params['vertices'] = vertices
        params['size_scale'] = 1.0
        annealer = tsp_draw.size_scale.Annealer(**params)
        begin = 2
        end = 5 
        true_move = np.concatenate([vertices[:begin],
                                    np.flip(vertices[begin : end + 1], axis = 0),
                                    vertices[end + 1 :]], axis = 0)
        annealer._make_move(begin, end)
        np.testing.assert_equal(true_move, annealer.vertices)

    def test_next(self):
        '''
        Vertices are:
        [0, 1]           [3, 1]  [4, 1]
        [0, 0]           [3, 0]  [4, 0]
        Start off with criss-cross diagonals on large spanse with size scale large enough to
        include large horizontals. Then correctly switch diagonals. Try to incorrectly switch
        diagonals and fail. Then incorrectly switch diagonals back.
        '''
        vertices = np.array([[0, 0], [3, 1], [4, 1], [4, 0], [3, 0], [0, 1]])
        true_vertices = vertices.copy()
        int_stack = [2, 1, 1, 2, 2, 1][::-1]
        crit_prob = np.exp(6 - 2 * np.sqrt(10))
        uniform_stack = [np.sqrt(crit_prob), 0.9 * crit_prob][::-1] 
        params = self.params.copy()
        params['vertices'] = vertices
        params['temperature'] = 1.0
        params['size_scale'] = 2.5 
        params['rand_state'] = fake_random.State(int_stack = int_stack, uniform_stack = uniform_stack)
        annealer = tsp_draw.size_scale.Annealer(**params)

        # First test that the size scale pool is right.
        np.testing.assert_equal(annealer.pool_v, np.array([0, 1, 4, 5]))

        # The large diagonals are un-criss crossed.
        next(annealer)
        true_vertices = true_vertices[[0, 4, 3, 2, 1, 5], :]
        np.testing.assert_equal(annealer.vertices, true_vertices)

        # Propose to criss-cross the large diagonals again (this results in larger energy).
        # The proposal should fail.
        next(annealer)
        np.testing.assert_equal(annealer.vertices, true_vertices)

        # Again propose to criss-cross the large diagonals. This time the proposal should
        # be accepted.
        next(annealer)
        true_vertices = true_vertices[[0, 4, 3, 2, 1, 5], :]
        np.testing.assert_equal(annealer.vertices, true_vertices)

if __name__ == '__main__':
    unittest.main() 
