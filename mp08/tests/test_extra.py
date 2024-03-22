import unittest, json, submitted
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

# Prisoner's dilemma with positive rewards if opponent cooperates
rewards = np.array([[[-1,2],[-2,1]],[[-1,-2],[2,1]]])

opponent_strategies = np.array([[[0,0],[0,0]],[[0,0],[0,1]],[[0,0],[1,0]],[[0,0],[1,1]],
                                [[0,1],[0,0]],[[0,1],[0,1]],[[0,1],[1,0]],[[0,1],[1,1]],
                                [[1,0],[0,0]],[[1,0],[0,1]],[[1,0],[1,0]],[[1,0],[1,1]],
                                [[1,1],[0,0]],[[1,1],[0,1]],[[1,1],[1,0]],[[1,1],[1,1]]])
# TestSequence
class TestStep(unittest.TestCase):
    @weight(10)
    def test_extra(self):
        score = 0
        played = 0
        for opponent_number in range(16):
            opponent_strategy = opponent_strategies[opponent_number,:,:]
            preva = 1
            prevb = 1
            nplays = 100
            for t in range(nplays):
                b = 1
                if np.random.uniform() < submitted.sequential_strategy[preva,prevb]:
                    b = 1
                else:
                    b = 0
                if np.random.uniform() < opponent_strategy[preva,prevb]:
                    a = 1
                else:
                    a = 0
                score += rewards[1,a,b]
                played += 1
                preva = a
                prevb = b
        print('You played %d games, against all 16 possible fixed-strategy opponents'%(played))
        print('and you won an average of %2f points per game'%(score/played))
        self.assertGreater(score/played,0.2,msg='That score is not enough to get extra credit!')
        print('Congratulations!  That score is enough for extra credit!')
