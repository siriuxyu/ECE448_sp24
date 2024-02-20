import torch
import torch.nn as nn


def create_sequential_layers():
    """
    Task: Create neural net layers using nn.Sequential.

    Requirements: Return an nn.Sequential object, which contains:
        1. a linear layer (fully connected) with 2 input features and 3 output features,
        2. a sigmoid activation layer,
        3. a linear layer with 3 input features and 5 output features.
    """
    new_layers = nn.Sequential(
        nn.Linear(2, 3),
        nn.Sigmoid(),
        nn.Linear(3, 5)
    )
    return new_layers
    # raise NotImplementedError("You need to write this part!")


def create_loss_function():
    """
    Task: Create a loss function using nn module.

    Requirements: Return a loss function from the nn module that is suitable for
    multi-class classification.
    """
    loss = nn.MSELoss()
    return loss
    # raise NotImplementedError("You need to write this part!")


class NeuralNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize your neural network here.
        """
        super().__init__()
        ################# Your Code Starts Here #################
        self.linear1 = nn.Linear(2883, 31)
        self.reLU = nn.ReLU()
        self.linear2 = nn.Linear(31, 5)
        
        self.loss = nn.CrossEntropyLoss()
        
        # raise NotImplementedError("You need to write this part!")
        ################## Your Code Ends here ##################

    def forward(self, x):
        """
        Perform a forward pass through your neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        ################# Your Code Starts Here #################
        x_temp = self.linear1(x)
        x_temp = self.reLU(x_temp)
        y_pred = self.linear2(x_temp)
        
        return y_pred
        ################## Your Code Ends here ##################


def train(train_dataloader, epochs):
    """
    The autograder will call this function and compute the accuracy of the returned model.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        test_dataloader:    a dataloader for the testing set and labels
        epochs:             the number of times to iterate over the training set

    Outputs:
        model:              trained model
    """

    ################# Your Code Starts Here #################
    """
    Implement backward propagation and gradient descent here.
    """
    # Create an instance of NeuralNet, a loss function, and an optimizer
    model = NeuralNet()
    loss_fn = model.loss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Train the model
    for epoch in range(epochs):
        for input, labels in train_dataloader:
            y_pred = model.forward(input)
            loss = loss_fn(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    return model

    
    # raise NotImplementedError("You need to write this part!")
    ################## Your Code Ends here ##################

        
