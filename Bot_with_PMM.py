import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import chess
b=chess.Board()
class Node:
    def __init__(self,board:chess.Board,context):
        self.board=board
        self.board_tensor=self._tens()
        self.context=context
    def _tens(self):
        tens=torch.zeros((32,4),dtype=torch.float32)
        i=0
        for square,piece in self.board.piece_map().items():
            piece_type=piece.piece_type-1
            color=int(piece.color)
            tens[i,0]=piece_type
            tens[i,1]=color
            tens[i,2]=square//8-3.5
            tens[i,3]=square%8-3.5
            i+=1
        return tens
def search
#input is shape batch,32,4
class ResBlock(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.attn=nn.MultiheadAttention(embed_dim=dim, num_heads=4, dropout=0.1, batch_first=True)
        self.fc=nn.Sequential(
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,256)
        )
    def forward(self,x):
        #x is batch 32 4
        batch_size=x.shape[0]
        attn_out,_=self.attn(x,x,x)
        attn_out=self.fc(attn_out.reshape(batch_size,256)).reshape(x.shape)
        x=x+attn_out
        return x
            
class chess_attention(nn.Module):
    def __init__(self, num_heads=4, dropout=0.1,encoder_dim=16):
        super().__init__()
        self.num_heads = num_heads
        self.encoder=nn.Linear(4, encoder_dim)
        self.attention = nn.MultiheadAttention(embed_dim=encoder_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.resblocks=nn.Sequential(
            *[ResBlock(encoder_dim) for _ in range(4)]
        )
        self.outputfc=nn.Sequential(
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,4)
        )
        
    def forward(self,x):
        batch_size=x.shape[0]
        assert x.shape[1]==32 
        assert x.shape[2]==4
        x=self.encoder(x.reshape(batch_size*32,4)).reshape(batch_size,32,16)#batch,seq,dim
        x,_ = self.attention(x, x, x)
        x=self.resblocks(x)
        x=self.outputfc(x.reshape(batch_size,256))
        return x #batch,4


if __name__ == "__main__":
    board = chess.Board()
    for _ in range(8):
        board.push(np.random.choice(list(board.legal_moves)))
    tensor = crunch_board(board,depth=8)
    print(tensor.shape)  # Should print torch.Size([1, 104, 8, 8])
    model = chess_attention(dropout=0.1,encoder_dim=16)
    output = model(tensor)
    print(output.shape)  # Should print torch.Size([1, 8, 13, 16])
