import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
from Search import *
from moves import pmoves
# Configuration
CONTEXT_LENGTH = 256*4  # total context vector size
CTX_TOKENS = 4   # number of context tokens (CTX_TOKENS * encoder_dim == CONTEXT_LENGTH)
MAX_PIECES = 33        # number of piece slots in board encoding
ENCODER_DIM = 256       # per-token embedding dim (must satisfy CTX_TOKENS * ENCODER_DIM == CONTEXT_LENGTH)

assert CTX_TOKENS * ENCODER_DIM == CONTEXT_LENGTH, "CTX_TOKENS * ENCODER_DIM must equal CONTEXT_LENGTH"


# ---- Small cross-attention updater (low param) ----
class SmallCrossAttnUpdater(nn.Module):
    def __init__(self,
                 ctx_dim=CONTEXT_LENGTH,
                 new_dim=256,
                 hidden_q=128,
                 hidden_kv=32,
                 M=4,
                 num_heads=8,
                 eps=1e-5):
        super().__init__()
        assert hidden_q % num_heads == 0
        self.ctx_dim = ctx_dim
        self.new_dim = new_dim
        self.hidden_q = hidden_q
        self.hidden_kv = hidden_kv
        self.M = M

        self.ctx_to_q = nn.Linear(ctx_dim, hidden_q, bias=True)
        self.new_to_kv = nn.Linear(new_dim, M * hidden_kv, bias=True)

        # Cross-attention: queries hidden_q, keys/values hidden_kv
        self.mha = nn.MultiheadAttention(embed_dim=hidden_q,
                                         num_heads=num_heads,
                                         kdim=hidden_kv,
                                         vdim=hidden_kv,
                                         batch_first=True,
                                         dropout=0.0)

        self.out_map = nn.Linear(hidden_q, ctx_dim, bias=True)

        # tiny learned per-dim gate
        self.gate = nn.Parameter(torch.zeros(ctx_dim))
        self.ln = nn.LayerNorm(ctx_dim, eps=eps)

    def forward(self, ctx, new, new_mask=None):
        """
        ctx: (B, ctx_dim)
        new: (B, new_dim)
        """
        B = ctx.shape[0]
        q = self.ctx_to_q(ctx).unsqueeze(1)                      # (B,1,hidden_q)
        kv = self.new_to_kv(new).view(B, self.M, self.hidden_kv) # (B,M,hidden_kv)
        attn_out, attn_weights = self.mha(query=q, key=kv, value=kv,
                                          key_padding_mask=new_mask)  # (B,1,hidden_q)
        attn_out = attn_out.squeeze(1)                           # (B, hidden_q)
        attn_mapped = self.out_map(attn_out)                     # (B, ctx_dim)

        g = torch.sigmoid(self.gate)                             # (ctx_dim,)
        updated = g * attn_mapped + (1.0 - g) * ctx              # (B, ctx_dim)
        updated = self.ln(updated)
        return updated, attn_weights

# ---- Transformer-ish blocks ----
class ResBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                          batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x, attn_mask=None):
        # x: (B, S, dim)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=attn_mask)
        x = x + attn_out
        x = self.ln1(x)
        f = self.ffn(x)
        x = x + f
        x = self.ln2(x)
        return x

# ---- Full model integrating Node ----
class ChessAttention(nn.Module):
    def __init__(self, num_heads=16, dropout=0.1, encoder_dim=ENCODER_DIM,
                 resblocks=16, ctx_tokens=CTX_TOKENS, context_length=CONTEXT_LENGTH,
                 new_dim=512):
        super().__init__()
        assert ctx_tokens * encoder_dim == context_length
        self.encoder = nn.Linear(4, encoder_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=encoder_dim,
                                               num_heads=num_heads,
                                               batch_first=True,
                                               dropout=dropout)
        self.resblocks = nn.ModuleList([ResBlock(encoder_dim, num_heads=num_heads, dropout=dropout)
                                        for _ in range(resblocks)])
        # project pooled token features to new vector (256)
        self.new_proj = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim * 2),
            nn.ReLU(),
            nn.Linear(encoder_dim * 2, new_dim)
        )
        # example prediction head (optional)
        self.pred_head = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        self.policy_head = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.ReLU(),
            nn.Linear(128, len(pmoves))
        )
        self.context_updater = SmallCrossAttnUpdater(ctx_dim=context_length,
                                                     new_dim=new_dim,
                                                     hidden_q=128,
                                                     hidden_kv=32,
                                                     M=4,
                                                     num_heads=8)
    def forward(self, x, context=None):
        """
        x: (B, MAX_PIECES, 4)
        context: (B, CONTEXT_LENGTH) or None
        returns: pred (B,4), updated_context (B,CONTEXT_LENGTH), attn_weights
        """
        B = x.shape[0]
        device = x.device
        dtype = x.dtype
        if context is None:
            context = torch.zeros(B, CONTEXT_LENGTH, device=device, dtype=dtype)
        assert x.shape[1] == MAX_PIECES and x.shape[2] == 4

        # encode piece tokens
        x = self.encoder(x.view(B * MAX_PIECES, 4)).view(B, MAX_PIECES, -1)  # (B,32,encoder_dim)

        # reshape context into tokens and concat
        ctx_tokens = context.view(B, CTX_TOKENS, -1)  # (B, ctx_tokens, encoder_dim)
        x = torch.cat([x, ctx_tokens], dim=1)         # (B, 32+ctx_tokens, encoder_dim)

        # self-attention + resblocks
        x, _ = self.self_attn(x, x, x)
        for rb in self.resblocks:
            x = rb(x)

        # produce new vector by pooling and projecting
        pooled = x.mean(dim=1)         # (B, encoder_dim)
        new = self.new_proj(pooled)    # (B, new_dim)

        # update context
        updated_ctx, attn_w = self.context_updater(context, new)

        # optional prediction
        pred = self.pred_head(pooled)
        pred=pred[0]
        value=torch.tanh(pred[0]).item()
        antivalue=torch.tanh(pred[2]).item()
        variance=torch.tanh(pred[1]).item()
        antivariance=torch.exp(pred[3]).item()
        policy=self.policy_head(pooled)
        return value, antivalue,variance,antivariance, policy.tolist(), updated_ctx
def evaluator(node:Node,model:ChessAttention):
    x= node.board_tensor.unsqueeze(0)  # (1, MAX_PIECES, 4)
    context= node.context.unsqueeze(0)  # (1, CONTEXT_LENGTH)
    value, antivalue,variance,antivariance, policy, updated_ctx = model(x, context)
    return value, antivalue, policy[0], variance, updated_ctx.squeeze(0)
    
# ---- Example usage ----
if __name__ == "__main__":
    

    model = ChessAttention()
    
    print("Total params:", sum(p.numel() for p in model.parameters()))
   
    
