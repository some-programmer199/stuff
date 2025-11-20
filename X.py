import math
import random
import torch
import chess
import moves  # assuming you have this with pmove_to_idx
import tqdm
import threading
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, Tuple, Optional, List, Any

# ========================================
# Worker-side evaluator (for multiprocessing)
# ========================================
_worker_evaluator = None

def _worker_init(eval_pickle_bytes):
    global _worker_evaluator
    try:
        _worker_evaluator = pickle.loads(eval_pickle_bytes)
    except Exception as e:
        print("Worker init failed:", e)
        _worker_evaluator = None

def _worker_eval(task):
    global _worker_evaluator
    fen, bt_np, ctx_np = task
    if _worker_evaluator is None:
        return fen, None
    try:
        import torch as _torch
        node = type('obj', (), {})()
        node.board_tensor = _torch.from_numpy(bt_np)
        node.context = _torch.from_numpy(ctx_np)
        res = _worker_evaluator(node)

        # Expected: (value, antivalue, raw_policy, variance, antivariance, ctx)
        value, antivalue, raw_policy, variance, antivariance, ctx = res

        # Build legal policy dict
        import chess as _chess
        board = _chess.Board(fen)
        legal_moves = list(board.legal_moves)
        policy = {}

        if isinstance(raw_policy, dict):
            legal_uci = {m.uci() for m in legal_moves}
            policy = {k: float(v) for k, v in raw_policy.items() if k in legal_uci}
        else:
            logits = []
            for m in legal_moves:
                idx = moves.pmove_to_idx.get(m.uci(), -1)
                if 0 <= idx < len(raw_policy):
                    logits.append(raw_policy[idx])
                else:
                    logits.append(float('-inf'))
            if logits:
                probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=0)
                for m, p in zip(legal_moves, probs.tolist()):
                    policy[m.uci()] = float(p)

        if not policy and legal_moves:
            policy = {m.uci(): 1.0 / len(legal_moves) for m in legal_moves}

        return fen, (policy, float(value), float(antivalue), float(variance), float(antivariance), ctx.cpu().numpy() if hasattr(ctx, 'cpu') else ctx)
    except Exception as e:
        return fen, None


# ========================================
# Constants (fallback if Model.py not present)
# ========================================
try:
    from Model import CONTEXT_LENGTH, MAX_PIECES
except Exception:
    print("Model.py not found → using defaults")
    CONTEXT_LENGTH = 256 * 32
    MAX_PIECES = 33


# ========================================
# Node: lightweight board + tensor encoding
# ========================================
class Node:
    def __init__(self, board: chess.Board, context: Optional[torch.Tensor] = None):
        self.board = board
        self.board_tensor = self._to_tensor()
        self.context = context.clone().detach() if context is not None else torch.zeros(CONTEXT_LENGTH, dtype=torch.float32)

    def _to_tensor(self) -> torch.Tensor:
        tens = torch.zeros((MAX_PIECES, 4), dtype=torch.float32)
        i = 0
        for square, piece in self.board.piece_map().items():
            if i >= MAX_PIECES - 1:
                break
            tens[i, 0] = piece.piece_type - 1
            tens[i, 1] = int(piece.color)
            tens[i, 2] = (square // 8) - 3.5
            tens[i, 3] = (square % 8) - 3.5
            i += 1
        tens[MAX_PIECES-1, 0] = 10
        tens[MAX_PIECES-1, 1:] = self.board.ply()
        return tens

    def to(self, device):
        self.board_tensor = self.board_tensor.to(device)
        self.context = self.context.to(device)
        return self


# ========================================
# MCTS Node with your beloved variance madness
# ========================================
class MCTSNode:
    def __init__(self, state: Node, parent: Optional['MCTSNode'], prior: float, move_uci: Optional[str], turn: chess.Color):
        self.state = state
        self.parent = parent
        self.move = move_uci
        self.turn = turn
        self.prior = float(prior)

        self.children: Dict[str, MCTSNode] = {}
        self.N = 0
        self.W = 0.0        # sum of values (from root's perspective)
        self.W_anti = 0.0   # sum of antivalues
        self.Q = 0.0
        self.antiQ = 0.0
        self.variance = 1.0
        self.antivariance = 1.0

        self.virtual_loss = 0
        self.is_expanded = False

        self.opponent_softmax_weights: Optional[torch.Tensor] = None  # cached for selection

    def recompute_stats_and_weights(self, root_turn: chess.Color):
        if self.N > 0:
            self.Q = self.W / self.N
            self.antiQ = self.W_anti / self.N

        if not self.children:
            self.opponent_softmax_weights = None
            return

        children = list(self.children.values())
        opponent_vals = []
        opponent_vars = []
        for ch in children:
            if ch.turn == root_turn:  # opponent to move at child
                opponent_vals.append(ch.antiQ)
                opponent_vars.append(max(ch.antivariance, 1e-8))
            else:
                opponent_vals.append(ch.Q)
                opponent_vars.append(max(ch.variance, 1e-8))

        vals = torch.tensor(opponent_vals, dtype=torch.float32)
        vars_t = torch.tensor(opponent_vars, dtype=torch.float32)
        weighted = vals / vars_t
        weights = torch.softmax(weighted, dim=0)
        self.opponent_softmax_weights = weights


# ========================================
# The Beast: Optimized MCTS
# ========================================
class MCTS:
    def __init__(self, evaluator=None, c_puct: float = 2.5, dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25, batch_size: int = 16, num_workers: int = 8):
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.batch_size = batch_size
        self.num_workers = max(1, num_workers)

        self.eval_cache: Dict[str, Tuple] = {}
        self.lock = threading.Lock()

        self._thread_pool = ThreadPoolExecutor(max_workers=self.num_workers)
        self._process_pool = None
        if evaluator is not None:
            try:
                blob = pickle.dumps(evaluator)
                self._process_pool = ProcessPoolExecutor(
                    max_workers=self.num_workers,
                    initializer=_worker_init,
                    initargs=(blob,)
                )
            except:
                print("Warning: Could not create process pool — falling back to threads")

    def _default_eval(self, node: Node) -> Tuple[dict, float, float, float, float, torch.Tensor]:
        legal = list(node.board.legal_moves)
        n = max(1, len(legal))
        policy = {m.uci(): 1.0 / n for m in legal}
        return policy, 0.0, 0.0, 1.0, 1.0, node.context

    def _eval_node(self, mnode: MCTSNode) -> Tuple[dict, float, float, float, float, torch.Tensor]:
        fen = mnode.state.board.fen()
        cached = self.eval_cache.get(fen)
        if cached:
            return cached

        board = mnode.state.board
        if board.is_game_over():
            result = board.result(claim_draw=True)
            if result == "1-0":
                v = 1.0 if board.turn == chess.BLACK else -1.0
            elif result == "0-1":
                v = 1.0 if board.turn == chess.WHITE else -1.0
            else:
                v = 0.0
            policy = {}
            self.eval_cache[fen] = (policy, v, -v, 0.0, 0.0, mnode.state.context)
            return self.eval_cache[fen]

        if self.evaluator is None:
            res = self._default_eval(mnode.state)
        else:
            try:
                raw = self.evaluator(mnode.state)
                if raw is None or not isinstance(raw, (tuple, list)) or len(raw) < 6:
                    res = self._default_eval(mnode.state)
                else:
                    v, av, pol_raw, var, avar, ctx = raw
                    # build legal policy
                    legal = list(board.legal_moves)
                    policy = {}
                    if isinstance(pol_raw, dict):
                        legal_set = {m.uci() for m in legal}
                        policy = {k: float(v) for k, v in pol_raw.items() if k in legal_set}
                    else:
                        logits = torch.full((len(legal),), float('-inf'))
                        for i, m in enumerate(legal):
                            idx = moves.pmove_to_idx.get(m.uci(), -1)
                            if 0 <= idx < len(pol_raw):
                                logits[i] = pol_raw[idx]
                        if logits.isfinite().any():
                            probs = torch.softmax(logits, dim=0)
                            for m, p in zip(legal, probs.tolist()):
                                policy[m.uci()] = p
                    if not policy:
                        policy = {m.uci(): 1.0/len(legal) for m in legal}
                    res = (policy, float(v), float(av), float(var), float(avar), ctx)
            except:
                res = self._default_eval(mnode.state)

        self.eval_cache[fen] = res
        return res

    def _select(self, root: MCTSNode) -> Tuple[MCTSNode, List[MCTSNode]]:
        node = root
        path = [node]
        root_turn = root.turn

        while node.is_expanded and node.children:
            children = list(node.children.values())
            if node.opponent_softmax_weights is None:
                node.recompute_stats_and_weights(root_turn)

            weights = node.opponent_softmax_weights
            parent_N = max(1, sum(ch.N for ch in children))

            best_score = -float('inf')
            best_child = None
            best_idx = -1

            for i, child in enumerate(children):
                # exploitation: weighted by how much opponent likes this node
                exploit_val = child.Q if child.turn == root_turn else child.antiQ
                exploitation = float(weights[i]) * exploit_val

                # exploration: scaled by our own uncertainty
                var_expl = child.variance if child.turn == root_turn else child.antivariance
                u = self.c_puct * child.prior * math.sqrt(parent_N) / (1 + child.N + child.virtual_loss)
                u *= math.sqrt(var_expl)  # or just * var_expl — your choice

                score = exploitation + u
                if score > best_score:
                    best_score = score
                    best_child = child
                    best_idx = i

            best_child.virtual_loss += 3  # virtual loss
            node = best_child
            path.append(node)

        return node, path

    def _expand(self, leaf: MCTSNode):
        policy, v, av, var, avar, ctx = self._eval_node(leaf)
        board = leaf.state.board

        for uci, p in policy.items():
            move = chess.Move.from_uci(uci)
            new_board = board.copy(stack=False)
            new_board.push(move)
            new_node = Node(new_board, ctx)
            child = MCTSNode(
                state=new_node,
                parent=leaf,
                prior=p,
                move_uci=uci,
                turn=not board.turn
            )
            leaf.children[uci] = child

        leaf.is_expanded = True
        leaf.variance = float(var)
        leaf.antivariance = float(avar)

        # initial backup values
        leaf.W = v
        leaf.W_anti = av
        leaf.N = 1
        leaf.Q = v
        leaf.antiQ = av

    def _backup(self, path: List[MCTSNode], value: float, antivalue: float):
        for node in reversed(path):
            node.N += 1
            node.W += value
            node.W_anti += antivalue
            node.virtual_loss = max(0, node.virtual_loss - 3)
            node.recompute_stats_and_weights(path[0].turn)

    def search(self, board: chess.Board, num_sims: int = 800, context: Optional[torch.Tensor] = None):
        root_state = Node(board, context)
        root = MCTSNode(root_state, None, 1.0, None, board.turn)

        # Root evaluation + Dirichlet noise
        policy, v, av, var, avar, ctx = self._eval_node(root)
        if self.dirichlet_alpha:
            moves = list(policy.keys())
            if moves:
                noise = torch.tensor([random.gammavariate(self.dirichlet_alpha, 1.0) for _ in moves])
                noise = noise / noise.sum()
                policy = {m: policy[m] * (1 - self.dirichlet_epsilon) + n.item() * self.dirichlet_epsilon
                          for m, n in zip(moves, noise)}

        # Expand root
        board = root.state.board
        for uci, p in policy.items():
            move = chess.Move.from_uci(uci)
            new_board = board.copy(stack=False)
            new_board.push(move)
            new_node = Node(new_board, ctx)
            child = MCTSNode(new_node, root, p, uci, not board.turn)
            root.children[uci] = child
        root.is_expanded = True

        for _ in tqdm.tqdm(range(num_sims), desc="MCTS sims"):
            leaf, path = self._select(root)
            if not leaf.is_expanded:
                self._expand(leaf)
                value = leaf.Q
                antivalue = leaf.antiQ
            else:
                # already expanded (rare race), just reuse
                value = leaf.Q
                antivalue = leaf.antiQ

            self._backup(path, value, antivalue)

        visits = {uci: child.N for uci, child in root.children.items()}
        best_uci = max(visits, key=visits.get)
        best_child = root.children[best_uci]

        return best_uci, visits, best_child

    def close(self):
        for pool in (self._thread_pool, self._process_pool):
            if pool:
                pool.shutdown(wait=True)


# ========================================
# Quick test (uncomment to run)
# ========================================
if __name__ == "__main__":
    board = chess.Board()
    mcts = MCTS(evaluator=None, num_sims=400)  # no model → uniform random
    move, visits, _ = mcts.search(board, num_sims=800)
    print("Best move:", move)
    print("Visit distribution:", {chess.Move.from_uci(uci).uci(): n for uci, n in visits.items()})
    mcts.close()
