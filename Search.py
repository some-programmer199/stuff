import math
import random
import torch
import chess
import moves
import tqdm
# Try to import constants from Model if available; fall back to sensible defaults
try:
    from Model import CONTEXT_LENGTH, MAX_PIECES
except Exception:
    print("Could not import CONTEXT_LENGTH and MAX_PIECES from Model.py; using default values.")
    CONTEXT_LENGTH = 256 * 32
    MAX_PIECES = 33
def softmax(x):
    e_x = torch.exp(x - torch.max(x))
    return e_x / e_x.sum()
# ---- Lightweight Node wrapper ----
class Node:
    """Container for a chess.Board and its fixed-size encoding/context."""
    def __init__(self, board: chess.Board,move:chess.Move, context: torch.Tensor = None, ctx_len: int = CONTEXT_LENGTH):
        self.board = board
        self.board_tensor = self._to_tensor()
        self.expanded = False
        self.children = []
        if context is None:
            self.context = torch.zeros(ctx_len, dtype=torch.float32)
        else:
            self.context = context.clone().detach().to(dtype=torch.float32)

    def _to_tensor(self) -> torch.Tensor:
        tens = torch.zeros((MAX_PIECES, 4), dtype=torch.float32)
        i = 0
        for square, piece in self.board.piece_map().items():
            if i >= MAX_PIECES:
                break
            piece_type = float(piece.piece_type - 1)
            color = float(int(piece.color))
            rank = float(square // 8) - 3.5
            file = float(square % 8) - 3.5
            tens[i, 0] = piece_type
            tens[i, 1] = color
            tens[i, 2] = rank
            tens[i, 3] = file
            i += 1
         
        tens[MAX_PIECES-1,:]=torch.tensor(float(self.board.ply()), dtype=torch.float32).repeat(4)
        return tens
    

    def to_device(self, device):
        self.board_tensor = self.board_tensor.to(device)
        self.context = self.context.to(device)
        return self

    def move(self, move_uci: str):
        move = chess.Move.from_uci(move_uci)
        self.board.push(move)
        self.board_tensor = self._to_tensor()

# ---- MCTS node with statistics ----
class MCTSNode:
    def __init__(self, state: Node, parent=None, prior: float = 0.0, move_uci: str = None):
        self.state = state
        self.parent = parent
        self.move = move_uci
        self.prior = float(prior)
        self.children = {}  # move_uci -> MCTSNode
        self.N = 0
        self.W = 0.0
        self.antiW = 0.0
        self.Q = 0.0
        self.antiQ = 0.0
        self.is_expanded = False

    def is_root(self):
        return self.parent is None

    def expand(self, policy: dict):
        for m, p in policy.items():
            if m in self.children:
                continue
            new_board = self.state.board.copy()
            try:
                new_board.push(chess.Move.from_uci(m))
            except Exception:
                continue
            child_node = Node(new_board,chess.Move.from_uci(m), context=self.state.context)
            self.children[m] = MCTSNode(child_node, parent=self, prior=p, move_uci=m)
        self.is_expanded = True

    def total_children_visits(self) -> int:
        return sum(ch.N for ch in self.children.values())

    def update_Q(self):
        # Softmax over antiQ for Q, and over Q for antiQ
        children = list(self.children.values())
        if not children:
            self.Q = 0.0
            self.antiQ = 0.0
            self.N = 0
            self.W = 0.0
            self.antiW = 0.0
            return
        Qs = torch.tensor([ch.Q for ch in children], dtype=torch.float32)
        antiQs = torch.tensor([ch.antiQ for ch in children], dtype=torch.float32)
        N = sum(ch.N for ch in children)
        W = sum(ch.W for ch in children)
        antiW = sum(ch.antiW for ch in children)
        # Softmax weights
        antiQ_weights = softmax(antiQs)
        Q_weights = softmax(Qs)
        self.Q = float((antiQ_weights * Qs).sum())
        self.antiQ = float((Q_weights * antiQs).sum())
        self.N = N
        self.W = W
        self.antiW = antiW

# ---- MCTS engine ----
class MCTS:
    def __init__(self, evaluator=None, c_puct: float = 1.2, dirichlet_alpha: float = None, epsilon: float = 0.25):
        """
        evaluator: callable(node: Node) -> (policy_dict, value_float, updated_context_tensor)
                   - policy_dict: {move_uci: prob}
                   - value_float: in [-1,1] from perspective of node.state.board.turn
        c_puct: exploration constant
        dirichlet_alpha/epsilon: optional root noise parameters
        """
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.epsilon = epsilon

    def _default_evaluate(self, node: Node):
        legal = list(node.board.legal_moves)
        pol = {m.uci(): 1.0 / max(1, len(legal)) for m in legal}
        return pol, 0.0, 0.0, node.context

    def _select(self, root: MCTSNode):
        path = [root]
        node = root
        while node.is_expanded and node.children:
            parent_N = max(1, node.N)
            best_score = -float('inf')
            best_child = None
            children = list(node.children.values())
            antiQs = torch.tensor([ch.antiQ for ch in children], dtype=torch.float32)
            antiQ_weights = softmax(antiQs)
            for i, (m, ch) in enumerate(node.children.items()):
                U = self.c_puct * ch.prior * math.sqrt(parent_N) / (1 + ch.N)
                Q = ch.Q * float(antiQ_weights[i])
                score = Q + U
                if score > best_score:
                    best_score = score
                    best_child = ch
            node = best_child
            path.append(node)
            if node.state.board.is_game_over():
                break
        return node, path

    def _backup(self, path, value: float, antivalue: float):
        # Propagate value and antivalue up the path
        for i, node in enumerate(reversed(path)):
            if i == 0:
                # Leaf node: set W/antiW directly
                node.W = value
                node.antiW = antivalue
                node.N = 1
            else:
                node.update_Q()

    def _evaluate_and_expand(self, mnode: MCTSNode):
        board = mnode.state.board
        if board.is_game_over():
            res = board.result()
            if res == '1-0':
                winner = chess.WHITE
            elif res == '0-1':
                winner = chess.BLACK
            else:
                value = 0.0
                antivalue = 0.0
                policy = {}
                return policy, value, antivalue, mnode.state.context
            value = 1.0 if winner == board.turn else -1.0
            antivalue = -value
            return {}, value, antivalue, mnode.state.context

        if self.evaluator is None:
            policy, value, antivalue, new_ctx = self._default_evaluate(mnode.state)
        else:
            res = self.evaluator(mnode.state)
            if res is None:
                policy, value, antivalue, new_ctx = self._default_evaluate(mnode.state)
            else:
                # Unpack model output
                value, antivalue, raw_policy, new_ctx = res
                # Map raw_policy (list/array of logits) to legal moves
                legal_moves = list(board.legal_moves)
                policy = {}
                # Softmax over only legal moves
                import torch
                logits = torch.tensor([raw_policy[moves.pmove_to_idx[m.uci()]] if m.uci() in moves.pmove_to_idx else float('-inf') for m in legal_moves], dtype=torch.float32)
                # Mask illegal moves (set to -inf)
                mask = torch.tensor([m.uci() in moves.pmove_to_idx for m in legal_moves], dtype=torch.bool)
                logits[~mask] = float('-inf')
                probs = torch.softmax(logits, dim=0)
                for i, m in enumerate(legal_moves):
                    if mask[i]:
                        policy[m.uci()] = float(probs[i])
                # If all probs are zero (shouldn't happen), fallback to uniform
                if not policy or sum(policy.values()) == 0.0:
                    policy = {m.uci(): 1.0 / len(legal_moves) for m in legal_moves}

        mnode.expand(policy)
        return policy, float(value), float(antivalue), new_ctx

    def search(self, root_state: Node, num_sims: int = 100):
        root = MCTSNode(root_state, parent=None, prior=1.0)
        policy, value, antivalue, _ = (self.evaluator(root_state) if self.evaluator else self._default_evaluate(root_state)) or ({}, 0.0, 0.0, None)
        if not isinstance(policy, dict):
            try:
                policy = { (m.uci() if hasattr(m,'uci') else str(m)): float(p) for m,p in policy }
            except Exception:
                policy = {}
        if policy:
            s = float(sum(policy.values()))
            if s <= 0:
                policy = {}
            else:
                for k in policy:
                    policy[k] = float(policy[k]) / s

        if policy:
            if self.dirichlet_alpha is not None and root_state.board.legal_moves:
                moves = list(policy.keys())
                noise = [random.gammavariate(self.dirichlet_alpha, 1.0) for _ in moves]
                s = sum(noise)
                noise = [n / s for n in noise]
                for i, m in enumerate(moves):
                    policy[m] = policy.get(m, 0.0) * (1 - self.epsilon) + noise[i] * self.epsilon
            root.expand(policy)
            root.is_expanded = True
        else:
            legal = list(root_state.board.legal_moves)
            pol = {m.uci(): 1.0 / max(1, len(legal)) for m in legal}
            root.expand(pol)
            root.is_expanded = True

        for _ in tqdm.tqdm(range(num_sims), desc="MCTS Simulations"):
            leaf, path = self._select(root)
            policy, value, antivalue, _ = self._evaluate_and_expand(leaf)
            self._backup(path, value, antivalue)
        best_child = max(root.children.values(), key=lambda ch: ch.N, default=None)
        best_move = best_child.move if best_child else None
        return best_child, best_move


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


def action_probs(root: MCTSNode, temp: float = 0.0):
    """Return a selected best move and the visit-count distribution from a completed root.
    If temp==0 -> return the argmax by visit count (deterministic). Otherwise return
    a probability distribution proportional to N^(1/temp).
    """
    visits = {m: ch.N for m, ch in root.children.items()}
    if not visits:
        return None, {}
    if temp == 0.0:
        best = max(visits.items(), key=lambda x: x[1])[0]
        probs = {m: 1.0 if m == best else 0.0 for m in visits}
        return best, probs
    # soft distribution from visits
    vals = []
    for m in visits:
        vals.append(visits[m])
    # raise to power 1/temp
    powered = [v ** (1.0 / temp) if v > 0 else 0.0 for v in vals]
    s = sum(powered)
    probs = {}
    for (m, _), p in zip(visits.items(), powered):
        probs[m] = (p / s) if s > 0 else 1.0 / len(visits)
    best = max(probs.items(), key=lambda x: x[1])[0]
    return best, probs

