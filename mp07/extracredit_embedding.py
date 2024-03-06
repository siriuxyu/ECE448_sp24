import torch, math, json
import numpy as np
import chess.lib.heuristics

DTYPE=torch.float32
DEVICE=torch.device("cpu")

PIECES = 'kqbnrp'
def embed_board(side, board, flags):
    '''
    Return a pytorch tensor embedding of a PyChess (side,board,flags).
    embedding is a 15x8x8 tensor.
    Each channel is 8x8 to match the size of the chess board.
    Channels 0:12 are one-hot encodings of piece locations: kqbnrp, white then black.
    Channel 12 all locations are 0 if it's white's turn, 1 if it's black's turn.
    Channel 13 is 1 at the location of any rook still eligible for castling.
    Channel 14 is 1 at any position where a piece could move to capture a pawn en passant.
    '''
    embedding = np.zeros((15,8,8))
    for player in [0,1]:
        for i, piece in enumerate(board[player]):
            piecenum = 6*player + PIECES.index(piece[2])
            embedding[piecenum][piece[0]-1][piece[1]-1] = 1
            
    if side: # it's black's turn to play
        embedding[12,:,:] = 1
    if flags[0][0]: # white rook 1 is still eligible to castle
        embedding[13,0,7] = 1
    if flags[0][1]: # white rook 2 is still eligible to castle
        embedding[13,7,7] = 1
    if flags[0][2]: # black rook 1 is still eligible to castle
        embedding[13,0,0] = 1
    if flags[0][3]: # black rook 2 is still eligible to castle
        embedding[13,7,0] = 1
    if flags[1] != None: # there is a pawn susceptible to enpassant if another piece moves here
        embedding[14, flags[1][0]-1, flags[1][1]-1] = 1
    return embedding

def unembed_board(embedding):
    'Reverse the process of embed_board.  Return side, board, flags.'
    board = ([],[])
    for player in [0,1]:
        for piecenum in range(len(PIECES)):
            for pos in np.argwhere(embedding[6*player+piecenum]):
                board[player].append([int(pos[0]+1),int(pos[1]+1),PIECES[piecenum]])
    side = (embedding[12,0,0] > 0)
    flags = [[],None]
    for pos in [[0,7],[7,7],[0,0],[7,0]]:
        flags[0].append(embedding[13,pos[0],pos[1]] > 0)
    enpassant = np.argwhere(embedding[13])
    if len(enpassant) > 0:
        flags[1] = [ int(enpassant[0,0]), int(enpassant[0,1]) ]
    return side, board, flags

def initialize_weights():
    '''
    weights = initialize_weights()
    Generates a weights matrix from a 2*6*8*8 input vector to a 1-dimensional output,
    with weights set up to exactly match the heuristic in chess.lib.heuristics.
    You do not need to use this function, but it's a good debugging step:
    if you initialize using these weights and do not train, you should get a winratio of eactly  0.5.
    If you initialize using these weights and then train a linear model, you should get a winratio
    very slightly higher than 0.5.
    '''
    weights = torch.zeros((1, 15*8*8))
    bias = torch.zeros(1)
    for side in range(2):
        for piece in range(6):
            for x in range(8):
                for y in range(8):
                    board = ([],[])
                    board[side].append([x+1,y+1,PIECES[piece]])
                    flattened = side*6*8*8 + piece*8*8 + x*8 + y
                    weights[0,flattened] = chess.lib.heuristics.evaluate(board)
    return weights

class ChessDataset(torch.utils.data.Dataset):
    '''
    This is a pytorch Dataset object, that returns (embedding, value):
    embedding = a chessboard embedding, from embed_board(side, board, flags)
    value = the corresponding value.
    These are read from a pre-computed training or test file.
    Usage:
    dataset = ChessDataset(filename)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        for x,y in dataloader:
            # Perform one iteration of training
    '''
    def __init__(self, filename, verbose=True, max_tokens=math.inf):
        'Load up  to max_tokens from filename into memory'
        embeddings = []
        values = []
        with open(filename) as f:
            for line in f:
                if len(embeddings) < max_tokens:
                    game = json.loads(line)
                    moves = game['movestr'].split()
                    side, board, flags = chess.lib.utils.initBoardVars()
                    for (encoded,value) in zip(moves, game['values']):
                        move = chess.lib.utils.decode(encoded)
                        side, board, flags = chess.lib.makeMove(side,board,move[0],move[1],flags,move[2])
                        embeddings.append(embed_board(side, board, flags))
                        values.append([value])
                        if verbose and len(values) % 10000 == 0:
                            print('%s loaded board number %d...'%(filename,len(values)))
        self.embeddings = torch.tensor(embeddings,dtype=DTYPE,device=DEVICE)
        self.values = torch.tensor(values,dtype=DTYPE,device=DEVICE)

    def __len__(self):
        'Return the number of boards in this dataset'
        return(self.embeddings.size()[0])
    
    def __getitem__(self, i):
        "Return the i'th datum"
        return self.embeddings[i], self.values[i]
    
