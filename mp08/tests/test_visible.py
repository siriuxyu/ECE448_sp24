import unittest, json, submitted
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

# TestSequence
class TestStep(unittest.TestCase):
    def setUp(self):
        with open('solution.json') as f:
            self.solution = json.load(f)
            
    @weight(25)
    def test_gradient_ascent(self):
        init = np.array(self.solution['init'])
        learningrate = float(self.solution['learningrate'])
        rewards = np.array(self.solution['rewards'])
        r_logits = np.array(self.solution['g_logits'])
        r_utilities = np.array(self.solution['g_utilities'])
        h_logits,h_utilities=submitted.episodic_game_gradient_ascent(init,rewards,2,learningrate)
        for player in range(2):
            self.assertAlmostEqual(r_logits[1,player], h_logits[1,player], places=1,
                                   msg='''
                                   episodic_game_gradient_ascent
                                   logits[1,:]=[%f,%f] but should be [%f,%f]
                                   '''%(h_logits[1,0],h_logits[1,1],r_logits[1,0],r_logits[1,1]))

    @weight(25)
    def test_corrected_ascent(self):
        init = np.array(self.solution['init'])
        learningrate = float(self.solution['learningrate'])
        rewards = np.array(self.solution['rewards'])
        r_logits = np.array(self.solution['c_logits'])
        r_utilities = np.array(self.solution['c_utilities'])
        h_logits,h_utilities=submitted.episodic_game_corrected_ascent(init,rewards,2,learningrate)
        for player in range(2):
            self.assertAlmostEqual(r_logits[1,player], h_logits[1,player], places=1,
                                   msg='''
                                   episodic_game_corrected_ascent
                                   logits[1,:]=[%f,%f] but should be [%f,%f]
                                   '''%(h_logits[1,0],h_logits[1,1],r_logits[1,0],r_logits[1,1]))

