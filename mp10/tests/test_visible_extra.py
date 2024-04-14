import unittest, json, utils, submitted
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

# TestSequence
class TestStep(unittest.TestCase):
    def _test_U(self, model_name):
        model_file = 'models/model_%s.json'%model_name
        solution_file = 'solution_%s.json'%model_name
        model = utils.load_MDP(model_file)

        with open(solution_file, 'r') as f:
            data = json.load(f)
        U_gt = np.array(data['utility_extra'])

        U = submitted.policy_evaluation(model)
        diff = np.abs(U - U_gt)
        expr = diff.max() < 1e-2
        subtest_name = 'Utility function'
        msg = 'Testing %s (%s): '%(model_file, subtest_name)
        ind = np.unravel_index(np.argmax(diff, axis=None), diff.shape)
        msg += 'The difference between your utility and the ground truth shoud be less than 0.01. However, your U[%d, %d] = %.3f, while the ground truth U_gt[%d, %d] = %.3f'%(ind[0], ind[1], U[ind], ind[0], ind[1], U_gt[ind])
        self.assertTrue(expr, msg)
            
    @weight(3)
    def test_small_U(self):
        self._test_U('small')
        
    @weight(3)
    def test_large_U(self):
        self._test_U('large')
