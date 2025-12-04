import torch
import torch.nn as nn
import torch.nn.functional as F
from moves import pmoves

# ---------------- Configuration ----------------
MAX_PIECES = 33        # rounded for nice token size
CTX_TOKENS = 4           # number of context tokens
ENCODER_DIM = 256        # per-token embedding dim
CONTEXT_LENGTH = CTX_TOKENS * ENCODER_DIM

assert CTX_TOKENS * ENCODER_DIM == CONTEXT_LENGTH

# ---------------- Small Cross-Attention Updater ----------------
class SmallCrossAttnUpdater(nn.Module):
    def __init__(self, ctx_dim=CONTEXT_LENGTH, new_dim=256, hidden_q=128, hidden_kv=32, M=4, num_heads=8, eps=1e-5):
        super().__init__()
        assert hidden_q % num_heads == 0
        self.M = M
        self.hidden_kv = hidden_kv

        self.ctx_to_q = nn.Linear(ctx_dim, hidden_q)
        self.new_to_kv = nn.Linear(new_dim, M * hidden_kv)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_q, num_heads=num_heads,
                                         kdim=hidden_kv, vdim=hidden_kv, batch_first=True)
        self.out_map = nn.Linear(hidden_q, ctx_dim)
        self.gate = nn.Parameter(torch.zeros(ctx_dim))
        self.ln = nn.LayerNorm(ctx_dim, eps=eps)

    def forward(self, ctx, new, new_mask=None):
        """
        ctx: (B, ctx_dim)
        new: (B, new_dim)
        """
        B = ctx.size(0)
        q = self.ctx_to_q(ctx).unsqueeze(1)                 # (B,1,hidden_q)
        kv = self.new_to_kv(new).view(B, self.M, self.hidden_kv)  # (B, M, hidden_kv)
        attn_out, attn_weights = self.mha(q, kv, kv, key_padding_mask=new_mask)
        attn_out = attn_out.squeeze(1)                      # (B, hidden_q)
        attn_mapped = self.out_map(attn_out)               # (B, ctx_dim)
        g = torch.sigmoid(self.gate)
        updated = g * attn_mapped + (1 - g) * ctx
        updated = self.ln(updated)
        return updated, attn_weights

# ---------------- Transformer-ish Residual Block ----------------
class ResBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x, attn_mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=attn_mask)
        x = self.ln1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x

# ---------------- Chess Attention Model ----------------
class ChessAttention(nn.Module):
    def __init__(self, num_heads=8, dropout=0.1, encoder_dim=ENCODER_DIM,
                 resblocks=8, ctx_tokens=CTX_TOKENS, context_length=CONTEXT_LENGTH, new_dim=512):
        super().__init__()
        self.encoder = nn.Linear(4, encoder_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=encoder_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.encoder2 = nn.Linear(encoder_dim, encoder_dim)
        self.resblocks = nn.ModuleList([ResBlock(encoder_dim, num_heads=num_heads, dropout=dropout) for _ in range(resblocks)])
        self.new_proj = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim * 2),
            nn.ReLU(),
            nn.Linear(encoder_dim * 2, new_dim)
        )
        self.pred_head = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        self.policy_head = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.ReLU(),
            nn.Linear(128, len(pmoves))
        )
        self.context_updater = SmallCrossAttnUpdater(ctx_dim=context_length, new_dim=new_dim,
                                                     hidden_q=128, hidden_kv=32, M=4, num_heads=8)

    def forward(self, x, context=None):
        B = x.size(0)
        device, dtype = x.device, x.dtype
        if context is None:
            context = torch.zeros(B, CONTEXT_LENGTH, device=device, dtype=dtype)
        assert x.shape[1] == MAX_PIECES and x.shape[2] == 4

        # Encode piece tokens
        x = self.encoder(x.view(B * MAX_PIECES, 4)).view(B, MAX_PIECES, -1)

        # Reshape context to tokens and concatenate
        ctx_tokens = context.view(B, CTX_TOKENS, -1)
        x = torch.cat([x, ctx_tokens], dim=1)  # (B, MAX_PIECES + CTX_TOKENS, encoder_dim)
        x = self.encoder2(x)                   # linear on each token

        # Self-attention + residual blocks
        x, _ = self.self_attn(x, x, x)
        for rb in self.resblocks:
            x = rb(x)

        # Pool tokens and project
        pooled = x.mean(dim=1)
        new = self.new_proj(pooled)

        # Update context
        updated_ctx, attn_w = self.context_updater(context, new)

        # Predictions
        pred = self.pred_head(pooled)
        value = torch.tanh(pred[:, 0])
        variance = torch.tanh(pred[:, 1])
        antivalue = torch.tanh(pred[:, 2])
        policy = self.policy_head(pooled)

        return value, antivalue, variance, policy, updated_ctx

# ---------------- Example Usage ----------------
if __name__ == "__main__":
    model = ChessAttention(resblocks=20)
    dummy_input = torch.randn(2, MAX_PIECES, 4)
    dummy_context = torch.randn(2, CONTEXT_LENGTH)
    value, antivalue, variance, policy, updated_ctx = model(dummy_input, dummy_context)
    print("Value:", value)
    print("Total params:", sum(p.numel() for p in model.parameters()))

