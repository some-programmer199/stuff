import torch
import torch.nn as nn
import torch.nn.functional as F
from moves import pmoves

MAX_PIECES = 33
CTX_TOKENS = 6
ENCODER_DIM = 448
CONTEXT_LENGTH = CTX_TOKENS * ENCODER_DIM
HISTORY_LEN = 32
HISTORY_DIM = 256
HISTORY_LAYERS = 2
HISTORY_HEADS = 8
HISTORY_PAD_IDX = 0

class HistoryBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout, mlp_ratio=4):
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

class MoveHistoryEncoder(nn.Module):
    def __init__(self, hist_len=HISTORY_LEN, hist_dim=HISTORY_DIM, num_heads=HISTORY_HEADS,
                 layers=HISTORY_LAYERS, dropout=0.05, pad_idx=HISTORY_PAD_IDX):
        super().__init__()
        self.hist_len = hist_len
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(len(pmoves), hist_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, hist_len, hist_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hist_dim))
        self.blocks = nn.ModuleList([
            HistoryBlock(hist_dim, num_heads, dropout) for _ in range(layers)
        ])
        self.ln_final = nn.LayerNorm(hist_dim, eps=1e-6)
        self.out_proj = nn.Linear(hist_dim, CONTEXT_LENGTH)

    def forward(self, move_history, padding_mask=None):
        assert move_history.dim() == 2, "move_history must be (B, N)"
        B, N = move_history.shape
        assert N == self.hist_len, f"expected history length {self.hist_len}, got {N}"
        if padding_mask is None:
            padding_mask = move_history.eq(self.pad_idx)
        assert padding_mask.shape == (B, N)

        tokens = self.embedding(move_history)
        tokens = tokens + self.pos_embed
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)

        cls_mask = torch.zeros((B, 1), dtype=padding_mask.dtype, device=padding_mask.device)
        attn_mask = torch.cat([cls_mask, padding_mask], dim=1)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        pooled = self.ln_final(x[:, 0])
        return self.out_proj(pooled)

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
                 mlp_ratio=4):
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
        self.history_encoder = MoveHistoryEncoder(
            hist_len=HISTORY_LEN,
            hist_dim=HISTORY_DIM,
            num_heads=HISTORY_HEADS,
            layers=HISTORY_LAYERS,
            dropout=dropout,
            pad_idx=HISTORY_PAD_IDX
        )
        self.ln_final = nn.LayerNorm(encoder_dim, eps=1e-6)

    def forward(self, x, move_history, move_history_mask=None):
        B = x.size(0)
        assert move_history.shape[0] == B, "move_history batch must match x"

        x = self.encoder(x.view(B * MAX_PIECES, 4)).view(B, MAX_PIECES, -1)
        context = self.history_encoder(move_history, padding_mask=move_history_mask)
        assert context.shape == (B, CONTEXT_LENGTH)
        ctx_tokens = context.view(B, CTX_TOKENS, -1)
        x = torch.cat([x, ctx_tokens], dim=1)
        
        x = x + self.pos_embed
        x = self.encoder2(x)
        x = self.self_attn(x, x, x)[0]
        
        for rb in self.resblocks:
            x = rb(x)
        
        x = self.ln_final(x)
        pooled = x.mean(dim=1)

        pred = self.pred_head(pooled)
        value = torch.tanh(pred[:, 0])
        variance = F.softplus(pred[:, 1])
        antivalue = torch.tanh(pred[:, 2])
        policy = self.policy_head(pooled)

        return value, antivalue, variance, policy, context

if __name__ == "__main__":
    model = ChessAttention(resblocks=18)
    dummy_input = torch.randn(2, MAX_PIECES, 4)
    dummy_history = torch.full((2, HISTORY_LEN), HISTORY_PAD_IDX, dtype=torch.long)
    dummy_history[0, -3:] = torch.tensor([1, 5, 20], dtype=torch.long)
    dummy_history[1, -5:] = torch.tensor([3, 7, 11, 15, 18], dtype=torch.long)
    dummy_history_mask = dummy_history.eq(HISTORY_PAD_IDX)
    value, antivalue, variance, policy, context = model(dummy_input, dummy_history, dummy_history_mask)
    print("Value shape:", value.shape)
    print("Antivalue shape:", antivalue.shape)
    print("Variance shape:", variance.shape)
    print("Policy shape:", policy.shape)
    print("Context shape:", context.shape)
    print("Total params:", sum(p.numel() for p in model.parameters()))
