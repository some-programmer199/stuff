import torch
import torch.nn as nn
import math
import numpy as np
import chess
b=chess.Board()
piece_to_plane = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}
plane_to_piece = {v: k for k, v in piece_to_plane.items()}

def crunch_board(boardx: chess.Board,depth=8):
    tensorend = torch.zeros((0, 8, 8), dtype=torch.float32)
    for x in range(depth):
        tensor = torch.zeros(13, 8, 8)
        for square in chess.SQUARES:
            piece = boardx.piece_at(square)
            if piece:
                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)
                base = 0 if piece.color == chess.WHITE else 6
                plane = base + piece_to_plane[piece.piece_type]
                tensor[plane, row, col] = 1.0
        tensor[12, :, :] = float(boardx.ply())
        tensorend = torch.cat((tensorend, tensor), dim=0)
        try:
            boardx.pop()
        except IndexError:
            pass
    return tensorend.view(size=(1, 13*depth, 8, 8))

class chess_attention(nn.Module):
    def __init__(self, num_heads=4, dropout=0.1,encoder_dim=16):
        super().__init__()
        self.num_heads = num_heads
        self.encoder = nn.Sequential(
            nn.Linear(66, 64),
            nn.ReLU(),
            nn.Linear(64, encoder_dim)
        )
        self.attention = nn.MultiheadAttention(embed_dim=encoder_dim, num_heads=num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self,board_tensor):
        board_tensor = board_tensor.view(board_tensor.size(0), -1,13,64) #shape (batch_size, depth, 13, 64)
        encoded=torch.zeros(board_tensor.size(0),board_tensor.size(1),13,16)
        for depth in range(board_tensor.size(1)):
            for board in range(board_tensor.size(2)):
                if board < 12:
                    d=board_tensor[depth,board,:,:][0].repeat(16)
                    
                else:
                    d=board_tensor[:,depth,board,:] #d shape (batch_size, 64)
                    piece=board
                    d=torch.cat(d,torch.tensor([piece],dtype=torch.float32),depth,dim=1)
                    
                print(d.shape)
                print(encoded.shape)
                encoded[:,depth,board,:]=d
        #encoded shape (batch_size, depth, 13, encoder_dim)
        encoded=self.attention(encoded,encoded,encoded)[0]
        encoded=self.dropout(encoded)
        return encoded



if __name__ == "__main__":
    board = chess.Board()
    for _ in range(8):
        board.push(np.random.choice(list(board.legal_moves)))
    tensor = crunch_board(board,depth=8)
    print(tensor.shape)  # Should print torch.Size([1, 104, 8, 8])
    model = chess_attention(dropout=0.1,encoder_dim=16)
    output = model(tensor)
    print(output.shape)  # Should print torch.Size([1, 8, 13, 16])
