import torch, random, math, json
import numpy as np
from extracredit_embedding import ChessDataset, initialize_weights

DTYPE=torch.float32
DEVICE=torch.device("cpu")

###########################################################################################
def trainmodel():

    model = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(in_features=8*8*15, out_features=1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    L1loss = torch.nn.L1Loss()

    # ... and if you do, this initialization might not be relevant any more ...
    model[1].weight.data = initialize_weights()
    model[1].bias.data = torch.zeros(1)

    # ... and you might want to put some code here to train your model:
    trainset = ChessDataset(filename='extracredit_train.txt')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True)
    validset = ChessDataset(filename='extracredit_validation.txt')
    validloader = torch.utils.data.DataLoader(validset, batch_size=5000, shuffle=True)
    
    for epoch in range(100):
        for x,y in trainloader:
            y_pred = model(x)
            loss = L1loss(y_pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            valid_x, valid_y = next(iter(validloader))
            validation_loss = torch.nn.functional.mse_loss(model(valid_x), valid_y)
            print(f"epoch {epoch} training_loss {loss.item()} validation_loss {validation_loss.item()}")
            

    # ... after which, you should save it as "model_ckpt.pkl":
    torch.save(model, 'model_ckpt.pkl')


###########################################################################################
if __name__=="__main__":
    trainmodel()
    
    