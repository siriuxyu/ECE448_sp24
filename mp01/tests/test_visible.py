import unittest, json, reader, submitted
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

# TestSequence
class TestStep(unittest.TestCase):
    def setUp(self):
        with open('solution.json') as f:
            self.solution = json.load(f)
            
    @weight(9)
    def test_joint(self):
        ref = np.array(self.solution['Pjoint'])
        Pcond = np.array(self.solution['Pcond'])
        P0 = np.array(self.solution['P0'])
        hyp = submitted.joint_distribution_of_word_counts(P0, Pcond)
        (M, N) = ref.shape
        self.assertEqual(len(hyp.shape), 2,
                         'joint_distribution_of_word_counts should return a 2-dimensional array')
        self.assertLessEqual(M, hyp.shape[0],
                             'joint_distribution_of_word_counts dimension 0 should be at least %d'%(M))
        self.assertLessEqual(N, hyp.shape[1],
                             'joint_distribution_of_word_counts dimension 1 should be at least %d'%(N))
        for m in range(M):
            for n in range(N):
                self.assertAlmostEqual(ref[m,n], hyp[m,n], places=2,
                                       msg='''
                                       joint_distribution_of_word_counts[%d,%d] should be %g, not %g
                                       '''%(m,n,ref[m,n],ref[m,n]))

    @weight(9)
    def test_marginal(self):
        ref = np.array(self.solution['P1'])
        texts, count = reader.loadDir('data',False,False,False)
        hyp = submitted.marginal_distribution_of_word_counts(texts, self.solution['word1joint'])
        N = len(ref)
        self.assertLessEqual(N, len(hyp),
                             '''
                             marginal_distribution_of_word_counts(texts,"%s") should have length at
                             least %d, instead it is %d
                             '''%(self.solution['word1joint'],N,len(hyp)))
        for n in range(N):
            self.assertAlmostEqual(ref[n], hyp[n], places=2,
                                   msg='''
                                   marginal_distribution_of_word_counts(texts,"%s"][%d] should=%g, not %g
                                   '''%(self.solution['word1joint'],n,ref[n],ref[n]))

    @weight(8)
    def test_cond(self):
        ref = np.array(self.solution['Pcond'])
        texts, count = reader.loadDir('data',False,False,False)
        hyp = submitted.conditional_distribution_of_word_counts(texts,
                                                                self.solution['word0joint'],
                                                                self.solution['word1joint'])
        (M,N) = ref.shape
        self.assertLessEqual(M, hyp.shape[0],
                             msg='''
                             conditional_distribution_of_word_counts dimension 0 should be %d, not %d
                             '''%(M,hyp.shape[0]))
        self.assertLessEqual(N, hyp.shape[1],
                             '''
                             conditional_distribution_of_word_counts dimension 1 should be %d, not %d
                             '''%(N,hyp.shape[1]))
        for m in range(M):
            for n in range(N):
                if not np.isnan(ref[m,n]):
                    self.assertAlmostEqual(ref[m,n], hyp[m,n], places=2,
                                           msg='''
                                           conditional_distribution_of_word_counts[%d,%d] 
                                           should be %g, not %g
                                           '''%(m,n,ref[m,n],ref[m,n]))
                
    @weight(8)
    def test_mean(self):
        ref = np.array(self.solution['mu'])
        Pathe = np.array(self.solution['Pathe'])
        hyp = submitted.mean_vector(Pathe)
        self.assertEqual(len(hyp), 2, msg='''mean_vector should return a 2-vector''')
        self.assertAlmostEqual(ref[0], hyp[0], places=2,
                               msg='''mean_vector[0] should=%g, not %g'''%(ref[0], hyp[0]))
        self.assertAlmostEqual(ref[1], hyp[1], places=2,
                               msg='''mean_vector[1] should=%g, not %g'''%(ref[1], hyp[1]))
        
                
    @weight(8)
    def test_covariance(self):
        ref = np.array(self.solution['Sigma'])
        Pathe = np.array(self.solution['Pathe'])
        mu = np.array(self.solution['mu'])
        hyp = submitted.covariance_matrix(Pathe, mu)
        self.assertEqual(len(hyp), 2, msg='''covariance_matrix should have 2 rows''')
        self.assertEqual(len(hyp[0]), 2, msg='''covariance_matrix should have 2 columns''')
        self.assertAlmostEqual(ref[0,1], hyp[0,1], places=2,
                               msg='''covariance_matrix[0,1] should=%g, not %g'''%(ref[0,0],hyp[0,0]))
        self.assertAlmostEqual(ref[1,0], hyp[1,0], places=2,
                               msg='''covariance_matrix[1,0] should=%g, not %g'''%(ref[0,0],hyp[0,0]))
        self.assertAlmostEqual(ref[0,0], hyp[0,0], places=2,
                               msg='''covariance_matrix[0,0] should=%g, not %g'''%(ref[0,0],hyp[0,0]))
        self.assertAlmostEqual(ref[1,1], hyp[1,1], places=2,
                               msg='''covariance_matrix[1,1] should=%g, not %g'''%(ref[1,1],hyp[1,1]))

    @weight(8)
    def test_distribution_of_function(self):
        ref = self.solution['Pz']
        Pathe = np.array(self.solution['Pathe'])
        def f(x0,x1):
            if x0<1 and x1 < 1:
                return "Zero"
            elif x0 < 2 and x1 < 2:
                return "Small"
            else:
                return "Big"
        hyp = submitted.distribution_of_a_function(Pathe, f)
        for k in ref.keys():
            self.assertIn(k, hyp, msg='''distribution_of_a_function should have key %s'''%(k))
            self.assertAlmostEqual(ref[k], hyp[k], places=2,
                                   msg='''distribution_of_a_function[%s] should=%g, not %g'''%(k, ref[k], hyp[k]))
        
