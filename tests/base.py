import unittest
import numpy as np
import sys
import fake_random
sys.path.append('..')
import tsp_draw.base

class TestBaseMethods(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        angles = np.linspace(0, 2 * np.pi, 8)[:-1]
        self.vertices = [[np.cos(3 * angle), np.sin(3 * angle)] for angle in angles]
        self.vertices = np.array(self.vertices)
        self.params = {'n_steps' : 3, 'vertices' : self.vertices, 'temperature' : 0.001,
                       'temp_cool' : 0.99, 'rand_state' : fake_random.State([])}

    def test_get_cycle(self):
        annealer = tsp_draw.base.Annealer(**self.params)
        true_cycle = np.concatenate([self.vertices, [self.vertices[0]]], axis = 0)
        test_cycle = annealer.get_cycle()
        np.testing.assert_equal(true_cycle, test_cycle)

    def test_get_energy(self):
        annealer = tsp_draw.base.Annealer(**self.params)
        cycle = annealer.get_cycle()
        diffs = cycle[1:] - cycle[:-1]
        true_energy = np.linalg.norm(diffs, axis = 1).sum()
        test_energy = annealer.get_energy()
        self.assertEqual(true_energy, test_energy)

    def test_update_state(self):
        annealer = tsp_draw.base.Annealer(**self.params)
        annealer._update_state()
        self.assertEqual(annealer.temperature, 
                         self.params['temperature'] * self.params['temp_cool'])  
        self.assertEqual(annealer.steps_processed, 1)

    def test_run_proposal_trial(self):
        uniform_results = np.linspace(0, 1.0, 10)
        uniform_stack = list(np.flip(uniform_results))
        params = self.params.copy()
        params['rand_state'] = fake_random.State(uniform_stack)
        annealer = tsp_draw.base.Annealer(**params)

        energy_diff = 0.5 * annealer.temperature 
        critical_val = np.exp(-energy_diff / annealer.temperature)
        test_trials = [annealer._run_proposal_trial(energy_diff) for _ in uniform_results] 
        true_trials = [prob < critical_val for prob in uniform_results] 
        self.assertEqual(test_trials, true_trials)

if __name__ == '__main__':
    unittest.main()
