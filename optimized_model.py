import torch
import torch.nn as nn
import torch.nn.functional as F
from moves import pmoves

MAX_PIECES = 33
CTX_TOKENS = 6
ENCODER_DIM = 448
CONTEXT_LENGTH = CTX_TOKENS * ENCODER_DIM

class SmallCrossAttnUpdater(nn.Module):
    def __init__(self, ctx_dim=CONTEXT_LENGTH, new_dim=768, hidden_q=384, hidden_kv=128, M=8, num_heads=8):
        super().__init__()
        self.M = M
        self.hidden_kv = hidden_kv
        self.ctx_to_q = nn.Linear(ctx_dim, hidden_q)
        self.new_to_kv = nn.Linear(new_dim, M * hidden_kv)
        self.mha = nn.MultiheadAttention(hidden_q, num_heads, kdim=hidden_kv, vdim=hidden_kv, batch_first=True)
        self.out_map = nn.Linear(hidden_q, ctx_dim)
        self.gate = nn.Parameter(torch.zeros(ctx_dim))
        self.ln = nn.LayerNorm(ctx_dim, eps=1e-6)

    def forward(self, ctx, new, new_mask=None):
        B = ctx.size(0)
        q = self.ctx_to_q(ctx).unsqueeze(1)
        kv = self.new_to_kv(new).view(B, self.M, self.hidden_kv)
        attn_out, _ = self.mha(q, kv, kv, key_padding_mask=new_mask)
        attn_mapped = self.out_map(attn_out.squeeze(1))
        return self.ln(torch.sigmoid(self.gate) * attn_mapped + (1 - torch.sigmoid(self.gate)) * ctx), _

class ResBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.05, mlp_ratio=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x, attn_mask=None):
        x = self.ln1(x + self.attn(x, x, x, key_padding_mask=attn_mask)[0])
        return self.ln2(x + self.ffn(x))

class ChessAttention(nn.Module):
    def __init__(self, num_heads=8, dropout=0.05, encoder_dim=ENCODER_DIM,
                 resblocks=18, ctx_tokens=CTX_TOKENS, context_length=CONTEXT_LENGTH, 
                 new_dim=768, mlp_ratio=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, encoder_dim // 2),
            nn.GELU(),
            nn.Linear(encoder_dim // 2, encoder_dim)
        )
        self.pos_embed = nn.Parameter(torch.randn(1, MAX_PIECES + ctx_tokens, encoder_dim) * 0.02)
        self.self_attn = nn.MultiheadAttention(encoder_dim, num_heads, batch_first=True, dropout=dropout)
        self.encoder2 = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim * 2),
            nn.GELU(),
            nn.Linear(encoder_dim * 2, encoder_dim)
        )
        self.resblocks = nn.ModuleList([ResBlock(encoder_dim, num_heads, dropout, mlp_ratio) 
                                        for _ in range(resblocks)])
        self.new_proj = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim * 2, new_dim)
        )
        self.pred_head = nn.Sequential(
            nn.Linear(encoder_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 3)
        )
        self.policy_head = nn.Sequential(
            nn.Linear(encoder_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, len(pmoves))
        )
        self.context_updater = SmallCrossAttnUpdater(context_length, new_dim, 384, 128, 8, num_heads)
        self.ln_final = nn.LayerNorm(encoder_dim, eps=1e-6)

    def forward(self, x, context=None):
        B = x.size(0)
        if context is None:
            context = torch.zeros(B, CONTEXT_LENGTH, device=x.device, dtype=x.dtype)

        x = self.encoder(x.view(B * MAX_PIECES, 4)).view(B, MAX_PIECES, -1)
        ctx_tokens = context.view(B, CTX_TOKENS, -1)
        x = torch.cat([x, ctx_tokens], dim=1)
        
        x = x + self.pos_embed
        x = self.encoder2(x)
        x = self.self_attn(x, x, x)[0]
        
        for rb in self.resblocks:
            x = rb(x)
        
        x = self.ln_final(x)
        pooled = x.mean(dim=1)
        new = self.new_proj(pooled)
        updated_ctx, _ = self.context_updater(context, new)

        pred = self.pred_head(pooled)
        value = torch.tanh(pred[:, 0])
        variance = F.softplus(pred[:, 1])
        antivalue = torch.tanh(pred[:, 2])
        policy = self.policy_head(pooled)

        return value, antivalue, variance, policy, updated_ctx

if __name__ == "__main__":
    model = ChessAttention(resblocks=18)
    dummy_input = torch.randn(2, MAX_PIECES, 4)
    dummy_context = torch.randn(2, CONTEXT_LENGTH)
    value, antivalue, variance, policy, updated_ctx = model(dummy_input, dummy_context)
    print("Value:", value)
    print("Total params:", sum(p.numel() for p in model.parameters()))
