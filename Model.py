# Model.py
import torch
import torch.nn as nn
import chess
from moves import pmoves  # assuming this exists and has pmove_to_idx

# Configuration
CONTEXT_LENGTH = 256 * 4
CTX_TOKENS = 4
MAX_PIECES = 33
ENCODER_DIM = 256

assert CTX_TOKENS * ENCODER_DIM == CONTEXT_LENGTH


class SmallCrossAttnUpdater(nn.Module):
    def __init__(self, ctx_dim=CONTEXT_LENGTH, new_dim=512, hidden_q=128, hidden_kv=32, M=4, num_heads=8):
        super().__init__()
        self.ctx_to_q = nn.Linear(ctx_dim, hidden_q, bias=True)
        self.new_to_kv = nn.Linear(new_dim, M * hidden_kv, bias=True)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_q, num_heads=num_heads,
                                         kdim=hidden_kv, vdim=hidden_kv, batch_first=True)
        self.out_map = nn.Linear(hidden_q, ctx_dim, bias=True)
        self.gate = nn.Parameter(torch.zeros(ctx_dim))
        self.ln = nn.LayerNorm(ctx_dim)

    def forward(self, ctx, new, new_mask=None):
        B = ctx.shape[0]
        q = self.ctx_to_q(ctx).unsqueeze(1)
        kv = self.new_to_kv(new).view(B, -1, self.mha.vdim)
        attn_out, attn_weights = self.mha(q, kv, kv, key_padding_mask=new_mask)
        attn_out = attn_out.squeeze(1)
        attn_mapped = self.out_map(attn_out)
        g = torch.sigmoid(self.gate)
        updated = g * attn_mapped + (1.0 - g) * ctx
        return self.ln(updated), attn_weights


class ResBlock(nn.Module):
    def __init__(self, dim, num_heads=16, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        y, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.ln1(x + y)
        y = self.ffn(x)
        return self.ln2(x + y)


class ChessAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, ENCODER_DIM)
        self.self_attn = nn.MultiheadAttention(ENCODER_DIM, 16, batch_first=True)
        self.resblocks = nn.ModuleList([ResBlock(ENCODER_DIM, 16) for _ in range(16)])
        self.new_proj = nn.Sequential(nn.Linear(ENCODER_DIM, 512), nn.ReLU(), nn.Linear(512, 512))
        self.value_head = nn.Linear(ENCODER_DIM, 4)           # [value, variance, antivalue, antivariance]
        self.policy_head = nn.Linear(ENCODER_DIM, len(pmoves))
        self.context_updater = SmallCrossAttnUpdater(new_dim=512)

    def forward(self, x, context=None):
        # x: (B, MAX_PIECES, 4), context: (B, CONTEXT_LENGTH)
        B = x.shape[0]
        device, dtype = x.device, x.dtype
        if context is None:
            context = torch.zeros(B, CONTEXT_LENGTH, device=device, dtype=dtype)

        x = self.encoder(x.view(B * MAX_PIECES, 4)).view(B, MAX_PIECES, -1)
        ctx_tokens = context.view(B, CTX_TOKENS, ENCODER_DIM)
        x = torch.cat([x, ctx_tokens], dim=1)

        x, _ = self.self_attn(x, x, x)
        for blk in self.resblocks:
            x = blk(x)

        pooled = x.mean(dim=1)
        new_vec = self.new_proj(pooled)

        updated_ctx, _ = self.context_updater(context, new_vec)

        head = self.value_head(pooled)
        value = torch.tanh(head[:, 0])
        antivalue = torch.tanh(head[:, 2])
        variance = torch.tanh(head[:, 1]).abs() + 1e-6
        antivariance = torch.exp(head[:, 3])

        policy_logits = self.policy_head(pooled)

        return value, antivalue, variance, antivariance, policy_logits, updated_ctx


def evaluator(node, model):
    x = node.board_tensor.unsqueeze(0).to(next(model.parameters()).device)
    ctx = node.context.unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        v, av, var, avar, pol, new_ctx = model(x, ctx)
    return v.item(), av.item(), pol[0].cpu().tolist(), var.item(), new_ctx.squeeze(0).cpu()
