import unittest, json, reader, extra
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

# TestSequence
class TestStep(unittest.TestCase):
    def setUp(self):
        with open('solution.json') as f:
            self.solution = json.load(f)
            
    @weight(5)
    def test_extra(self):
        ref = np.array(self.solution['PY'])
        ref_p = self.solution['p']
        Pa = np.array(self.solution['Pa'])
        hyp_p, hyp = extra.estimate_geometric(Pa)
        self.assertAlmostEqual(ref_p, hyp_p, places=2,
                               msg='p should be %g not %g'%(ref_p,hyp_p))
        self.assertEqual(len(ref), len(hyp), 
                         msg='PY should have length %d not %d'%(len(ref),len(hyp)))
        self.assertAlmostEqual(ref[0],hyp[0],places=2,
                               msg='PY[0] should be %g not %g'%(ref[0],hyp[0]))
