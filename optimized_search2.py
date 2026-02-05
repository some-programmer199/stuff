import torch, chess, math, random
import numpy as np
from Model import ChessAttention
import moves

# ---------------- Config ----------------
MAX_NODES = 500_000
MAX_CHILDREN = 256
CONTEXT_LENGTH = 2688
MAX_PIECES = 33
C_PUCT = 2.0
VIRTUAL_LOSS = 3.0
TEMPERATURE = 2.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Node Tables ----------------
N      = torch.zeros(MAX_NODES, dtype=torch.int32)
W      = torch.zeros(MAX_NODES, dtype=torch.float32)
Wanti  = torch.zeros(MAX_NODES, dtype=torch.float32)
Q      = torch.zeros(MAX_NODES, dtype=torch.float32)
antiQ  = torch.zeros(MAX_NODES, dtype=torch.float32)
VAR    = torch.ones(MAX_NODES, dtype=torch.float32)
PRIOR  = torch.zeros(MAX_NODES, dtype=torch.float32)
TURN   = torch.zeros(MAX_NODES, dtype=torch.bool)
VLOSS  = torch.zeros(MAX_NODES, dtype=torch.int16)
EXP    = torch.zeros(MAX_NODES, dtype=torch.bool)

PARENT = torch.full((MAX_NODES,), -1, dtype=torch.int32)
CH_PTR = torch.zeros(MAX_NODES, dtype=torch.int32)
CH_LEN = torch.zeros(MAX_NODES, dtype=torch.int16)
CH_BUF = torch.full((MAX_NODES*8,), -1, dtype=torch.int32)
QW_BUF = torch.zeros(MAX_NODES*8, dtype=torch.float32)  # softmax weights

CTX   = torch.zeros((MAX_NODES, CONTEXT_LENGTH), dtype=torch.float32)
BOARD = torch.zeros((MAX_NODES, MAX_PIECES, 4), dtype=torch.float32)

# Python board objects for expansion only
BOARD_OBJ = {}

node_top = 0
child_top = 0

# ---------------- Helpers ----------------
def alloc_node():
    global node_top
    nid = node_top
    node_top += 1
    return nid

def alloc_children(k):
    global child_top
    ptr = child_top
    child_top += k
    return ptr

def board_to_tensor(board):
    tens = torch.zeros((MAX_PIECES,4))
    for i,(sq,pc) in enumerate(board.piece_map().items()):
        if i>=MAX_PIECES-1: break
        tens[i,0] = pc.piece_type-1
        tens[i,1] = float(pc.color)
        tens[i,2] = (sq>>3)-3.5
        tens[i,3] = (sq&7)-3.5
    tens[MAX_PIECES-1,0] = 10.0
    tens[MAX_PIECES-1,1] = float(board.ply())
    return tens

# ---------------- Tree Policy ----------------
def select_path(root):
    path = [root]
    nid = root
    while EXP[nid] and CH_LEN[nid] > 0:
        ptr = CH_PTR[nid]
        k = CH_LEN[nid]
        kids = CH_BUF[ptr:ptr+k]

        n = N[kids] + VLOSS[kids]
        w = W[kids] - VIRTUAL_LOSS*VLOSS[kids]
        q = w / torch.clamp(n, min=1)

        # antiQ
        aq = Wanti[kids] / torch.clamp(n, min=1)

        # softmax weighted blend per child
        weights = QW_BUF[ptr:ptr+k]
        values = torch.where(TURN[kids]==TURN[nid], q, aq)
        exploit = weights * values

        # UCT
        sqrt_parent = math.sqrt(N[nid]+1)
        u = C_PUCT * PRIOR[kids] * sqrt_parent / (1+n) * (1 + VAR[kids])

        scores = exploit + u

        idx = torch.argmax(scores).item()
        nid = kids[idx].item()
        VLOSS[nid] += 1
        path.append(nid)
    return nid, path

# ---------------- Expansion ----------------
def expand(nid, policy, new_ctx, var, softmax_weights):
    if EXP[nid]: return
    board = BOARD_OBJ[nid]
    legal = list(board.legal_moves)
    if not legal:
        EXP[nid] = True
        return

    ptr = alloc_children(len(legal))
    CH_PTR[nid] = ptr
    CH_LEN[nid] = len(legal)

    for i, mv in enumerate(legal):
        cid = alloc_node()
        CH_BUF[ptr+i] = cid

        nb = board.copy(stack=False)
        nb.push(mv)
        BOARD_OBJ[cid] = nb
        BOARD[cid] = board_to_tensor(nb)
        CTX[cid] = new_ctx

        PARENT[cid] = nid
        TURN[cid] = not board.turn
        PRIOR[cid] = policy.get(mv.uci(),0.0)
        QW_BUF[ptr+i] = softmax_weights.get(mv.uci(),1.0)

    EXP[nid] = True
    VAR[nid] = var

# ---------------- Backup ----------------
def backup(path, leaf, v, av, var):
    P = torch.tensor(path, dtype=torch.long)
    N[P] += 1
    same = TURN[P]==TURN[leaf]
    W[P]     += torch.where(same,v,av)
    Wanti[P] += torch.where(same,av,v)
    Q[P]     = W[P]/N[P]
    antiQ[P] = Wanti[P]/N[P]
    VAR[P]   = var
    VLOSS[P] = torch.clamp(VLOSS[P]-1,min=0)

# ---------------- Evaluate ----------------
def eval_batch(model, leaves):
    tens = BOARD[leaves].to(device)
    ctxs = CTX[leaves].to(device)
    with torch.no_grad():
        v, av, var, logits, new_ctxs, weights = model(tens, ctxs)  # network returns opponent softmax
    results = []
    for i, nid in enumerate(leaves):
        board = BOARD_OBJ[nid]
        legal = list(board.legal_moves)
        policy = {}
        softmax_weights = {}
        if legal:
            idxs = [moves.pmove_to_idx.get(m.uci(),-1) for m in legal]
            l = torch.full((len(legal),),-1e9,device=device)
            for j, idx in enumerate(idxs):
                if 0 <= idx < logits.size(1):
                    l[j] = logits[i,idx]
            probs = torch.softmax(l / TEMPERATURE, 0)
            policy = {m.uci(): float(p) for m,p in zip(legal,probs)}
            softmax_weights = {m.uci(): float(weights[i,j]) for j,m in enumerate(legal)}
        results.append((nid, policy, float(v[i]), float(av[i]), float(var[i]), new_ctxs[i].cpu(), softmax_weights))
    return results

# ---------------- MCTS Loop ----------------
def run_mcts(root, model, sims=800):
    for _ in range(sims):
        leaf, path = select_path(root)
        nid, policy, v, av, var, ctx, softmax_weights = eval_batch(model,[leaf])[0]
        CTX[nid] = ctx
        expand(nid, policy, ctx, var, softmax_weights)
        backup(path, nid, v, av, var)

    ptr = CH_PTR[root]
    k = CH_LEN[root]
    kids = CH_BUF[ptr:ptr+k]
    visits = N[kids]
    best = kids[torch.argmax(visits)].item()
    return best, visits, kids

# ---------------- Example Usage ----------------
board = chess.Board()
root = alloc_node()
BOARD_OBJ[root] = board
BOARD[root] = board_to_tensor(board)
CTX[root].zero_()
TURN[root] = board.turn
PRIOR[root] = 1.0

model = ChessAttention().to(device).eval()
best, visits, kids = run_mcts(root, model, sims=512)
print("Best Move:", BOARD_OBJ[best])
