'''
This is one of the modules you'll submit to the autograder. The positional encoding is implemented in this file.
'''

'''
Note:
Please do not modify any variable name given to you for code completion, especially those that have trainable parameters in torch
'''

import torch
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    '''
    We implement the positional encoding as alternating sin and cosine functions.
    In the __init__ function, you simply set the variable pe according the original paper.
    In the forward function, you add the pe to the input
    '''
    def __init__(self, d_model, max_seq_length):
        '''
        Initialize the positional encoding as alternating sin and cosine functions of the dimension position and the model size d_model
        Input:
            d_model (int) - dimension of the Transformer
            
            max_seq_length (int) - maximum sequence length of the Transformer
        '''
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)

        ## fill in the following code for positional encodings
        ## note that for better numerical accuracy, (1/10000) ^ (2i / d_model) = exp(2i * (-log(10000) /d_model))

        ##### YOUR CODE STARTS HERE #####





        ##### YOUR CODE ENDS HERE #####

        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """Add positional encoding (after appropriate slicing) to the input x

        Input:
            x (torch.Tensor) - Tensor of size B x T x d_model.

        Output:
            torch.Tensor - input with positional encoding added

        """
        ##### YOUR CODE STARTS HERE #####

        ##### YOUR CODE ENDS HERE #####