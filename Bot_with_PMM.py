import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import chess

CONTEXT_LENGTH=1024

b=chess.Board()

class SmallCrossAttnUpdater(nn.Module):
    """
    Update a 1024-d context vector from a 256-d new vector without a raw (1280->1024) linear.
    - ctx: (B, ctx_dim)  e.g. ctx_dim=1024
    - new: (B, new_dim)  e.g. new_dim=256
    Returns:
    - updated_ctx: (B, ctx_dim)
    - attn_weights: (B, 1, M)
    """
    def __init__(self,
                 ctx_dim=1024,
                 new_dim=256,
                 hidden_q=128,    # small query space
                 hidden_kv=32,    # small key/value dim per token
                 M=4,             # produce M pseudo-tokens from new
                 num_heads=8,
                 eps=1e-5):
        super().__init__()
        assert hidden_q % num_heads == 0, "hidden_q must be divisible by num_heads"

        self.ctx_dim = ctx_dim
        self.new_dim = new_dim
        self.hidden_q = hidden_q
        self.hidden_kv = hidden_kv
        self.M = M

        # project context down to small query space
        self.ctx_to_q = nn.Linear(ctx_dim, hidden_q, bias=True)

        # expand new vector -> M key/value tokens of dim hidden_kv
        self.new_to_kv = nn.Linear(new_dim, M * hidden_kv, bias=True)

        # MHA: queries have embed_dim = hidden_q; keys/values have dim = hidden_kv
        # batch_first=True so inputs are (B, S, D)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_q,
                                         num_heads=num_heads,
                                         kdim=hidden_kv,
                                         vdim=hidden_kv,
                                         batch_first=True,
                                         dropout=0.0)

        # map attended hidden_q back to ctx_dim
        self.out_map = nn.Linear(hidden_q, ctx_dim, bias=True)

        # tiny gating: a learned per-dimension bias vector that controls mixing
        # gate = sigmoid(self.gate) broadcast across batch
        self.gate = nn.Parameter(torch.zeros(ctx_dim))

        # a light layernorm for stability
        self.ln = nn.LayerNorm(ctx_dim, eps=eps)

    def forward(self, ctx, new, new_mask=None):
        """
        ctx: (B, ctx_dim)
        new: (B, new_dim)
        new_mask: optional key_padding_mask for M tokens (B, M) with True for padded positions
        """
        B = ctx.shape[0]
        # ctx -> queries
        q = self.ctx_to_q(ctx)           # (B, hidden_q)
        q = q.unsqueeze(1)               # (B, 1, hidden_q)  length=1 sequence

        # new -> M key/value tokens
        kv = self.new_to_kv(new)         # (B, M * hidden_kv)
        kv = kv.view(B, self.M, self.hidden_kv)  # (B, M, hidden_kv)

        # cross-attend: queries=context, keys/values=new_tokens
        attn_out, attn_weights = self.mha(query=q, key=kv, value=kv,
                                          key_padding_mask=new_mask)  # attn_out: (B,1,hidden_q)

        attn_out = attn_out.squeeze(1)   # (B, hidden_q)
        attn_mapped = self.out_map(attn_out)  # (B, ctx_dim)

        # blend using tiny per-dim gate (learned, input-independent)
        g = torch.sigmoid(self.gate)     # (ctx_dim,)
        updated = g * attn_mapped + (1.0 - g) * ctx  # (B, ctx_dim)

        updated = self.ln(updated)
        return updated, attn_weights
    

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
        self.context_updater=SmallCrossAttnUpdater(ctx_dim=CONTEXT_LENGTH,new_dim=256,hidden_q=128,hidden_kv=32,M=4,num_heads=8)
    def forward(self,x,context=torch.zeros(1,CONTEXT_LENGTH)):
        batch_size=x.shape[0]
        assert x.shape[1]==32 
        assert x.shape[2]==4
        x=self.encoder(x.reshape(batch_size*32,4)).reshape(batch_size,32,16)#batch,seq,dim
        x=torch.cat([x,context.reshape(batch_size,64,16)],dim=1) #batch,96,dim
        x,_ = self.attention(x, x, x)
        x=self.resblocks(x)
        x=self.outputfc(x.reshape(batch_size,256))
        context, _ = self.context_updater(context, x)
        return x, context #batch,4


if __name__ == "__main__":
    board = chess.Board()
    model=chess_attention()
    for i in model.named_parameters():
        print(i[0],i[1].shape.numel())
    print("Total params:", sum(p.numel() for p in model.parameters()))
