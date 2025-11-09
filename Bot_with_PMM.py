import torch
import torch.nn as nn
import chess

# Configuration
CONTEXT_LENGTH = 4096  # total context vector size
CTX_TOKENS = 32         # number of context tokens (CTX_TOKENS * encoder_dim == CONTEXT_LENGTH)
MAX_PIECES = 32         # number of piece slots in board encoding
ENCODER_DIM = 128       # per-token embedding dim (must satisfy CTX_TOKENS * ENCODER_DIM == CONTEXT_LENGTH)

assert CTX_TOKENS * ENCODER_DIM == CONTEXT_LENGTH, "CTX_TOKENS * ENCODER_DIM must equal CONTEXT_LENGTH"

# ---- Node helper (restores your Node bits) ----
class Node:
    """
    Lightweight container for a chess.Board and its context vector.

    - board: chess.Board
    - board_tensor: (MAX_PIECES, 4) float tensor
    - context: (CONTEXT_LENGTH,) float tensor
    """
    def __init__(self, board: chess.Board, context: torch.Tensor = None, ctx_len: int = CONTEXT_LENGTH):
        self.board = board
        self.board_tensor = self._to_tensor()
        if context is None:
            self.context = torch.zeros(ctx_len, dtype=torch.float32)
        else:
            self.context = context.clone().detach().to(dtype=torch.float32)

    def _to_tensor(self) -> torch.Tensor:
        """
        Encode pieces into a fixed (MAX_PIECES, 4) tensor:
          col 0: piece_type (0..5) or 0 for empty
          col 1: color (0 or 1) or 0
          col 2: rank centered (-3.5..3.5)
          col 3: file centered (-3.5..3.5)
        Remaining rows are zeros.
        """
        tens = torch.zeros((MAX_PIECES, 4), dtype=torch.float32)
        i = 0
        for square, piece in self.board.piece_map().items():
            if i >= MAX_PIECES:
                break
            piece_type = float(piece.piece_type - 1)  # 0..5
            color = float(int(piece.color))           # 0 or 1
            rank = float(square // 8) - 3.5
            file = float(square % 8) - 3.5
            tens[i, 0] = piece_type
            tens[i, 1] = color
            tens[i, 2] = rank
            tens[i, 3] = file
            i += 1
        return tens

    def to_device(self, device):
        self.board_tensor = self.board_tensor.to(device)
        self.context = self.context.to(device)
        return self

def nodes_to_batch(nodes, device=None, dtype=torch.float32):
    """
    Convert a list of Node -> (x, context) ready for model.forward:
      x: (B, MAX_PIECES, 4)
      context: (B, CONTEXT_LENGTH)
    """
    B = len(nodes)
    x = torch.stack([n.board_tensor for n in nodes], dim=0).to(dtype=dtype)
    ctx = torch.stack([n.context for n in nodes], dim=0).to(dtype=dtype)
    if device is not None:
        x = x.to(device)
        ctx = ctx.to(device)
    return x, ctx

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
    def __init__(self, num_heads=8, dropout=0.1, encoder_dim=ENCODER_DIM,
                 resblocks=32, ctx_tokens=CTX_TOKENS, context_length=CONTEXT_LENGTH,
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
            nn.Linear(encoder_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
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

        return pred, updated_ctx, attn_w

# ---- Example usage ----
if __name__ == "__main__":
    # Build two sample Nodes from fresh boards
    nodes = [Node(chess.Board()), Node(chess.Board())]
    x_batch, ctx_batch = nodes_to_batch(nodes)  # (B,32,4), (B,1024)

    model = ChessAttention()
    pred, updated_ctx, attn_w = model(x_batch, ctx_batch)

    print("pred.shape:", pred.shape)
    print("updated_ctx.shape:", updated_ctx.shape)
    print("attn_w.shape:", attn_w.shape)
    print("Total params:", sum(p.numel() for p in model.parameters()))
