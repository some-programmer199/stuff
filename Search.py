import math
import random
import torch
import chess
import moves
import tqdm
import threading
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# worker-side evaluator placeholder (set by initializer)
_worker_evaluator = None

def _worker_init(eval_pickle_bytes):
    """Initializer run in each worker process: restore evaluator callable."""
    global _worker_evaluator
    try:
        _worker_evaluator = pickle.loads(eval_pickle_bytes)
        # if the evaluator comes from Model module, ensure worker has a model instance
        try:
            modname = getattr(_worker_evaluator, '__module__', None)
            if modname == 'Model':
                import importlib
                M = importlib.import_module('Model')
                if not hasattr(M, 'model'):
                    # create a model instance in the worker (may be CPU)
                    try:
                        M.model = M.ChessAttention()
                    except Exception:
                        pass
                # prefer using Model.evaluator if available
                if hasattr(M, 'evaluator'):
                    _worker_evaluator = M.evaluator
        except Exception:
            pass
    except Exception:
        # last resort: try to import Model.evaluator if available
        try:
            from Model import evaluator as _e
            _worker_evaluator = _e
        except Exception:
            _worker_evaluator = None

def _worker_eval(task):
    """Worker evaluation entrypoint. task=(fen, board_tensor_np, ctx_np)
    Returns (fen, result_tuple) where result_tuple is the evaluator return.
    """
    global _worker_evaluator
    fen, bt_np, ctx_np = task
    # create minimal node-like object expected by evaluator
    try:
        import types
        node = types.SimpleNamespace()
        import torch as _torch
        node.board_tensor = _torch.from_numpy(bt_np)
        node.context = _torch.from_numpy(ctx_np)
        # call evaluator if available
        if _worker_evaluator is None:
            return fen, None
        res = _worker_evaluator(node)
        # normalize into (policy, value, antivalue, variance, antivariance, ctx)
        # worker may return (value, antivalue, raw_policy, variance, antivariance, ctx)
        try:
            value, antivalue, raw_policy, variance, antivariance, ctx = res
            # build policy dict
            import moves as _moves
            import chess as _chess
            legal = []
            # fen does not include side to move? it does; rebuild board
            try:
                b = _chess.Board(fen)
                legal = list(b.legal_moves)
            except Exception:
                legal = []
            policy = {}
            if isinstance(raw_policy, dict):
                legal_uci = {m.uci() for m in legal}
                policy = {k: float(v) for k, v in raw_policy.items() if k in legal_uci}
            else:
                if legal:
                    logits = []
                    for m in legal:
                        idx = _moves.pmove_to_idx.get(m.uci(), -1)
                        if idx >= 0 and idx < len(raw_policy):
                            logits.append(raw_policy[idx])
                        else:
                            logits.append(float('-inf'))
                    _logits = _torch.tensor(logits, dtype=_torch.float32)
                    mask = ~(_logits == float('-inf'))
                    if _logits.numel() > 0:
                        _logits[~mask] = float('-inf')
                        probs = _torch.softmax(_logits, dim=0)
                        for i, m in enumerate(legal):
                            if mask[i]:
                                policy[m.uci()] = float(probs[i])
            if not policy and legal:
                policy = {m.uci(): 1.0 / len(legal) for m in legal}
            return fen, (policy, float(value), float(antivalue), float(variance), float(antivariance), ctx)
        except Exception:
            return fen, None
    except Exception:
        return fen, None
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
         
        tens[MAX_PIECES-1,:]=torch.tensor([10,*[self.board.ply() for i in range(3)],],dtype=torch.float32)
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
    def __init__(self, state: 'Node' = None, parent=None, prior: float = 0.0, move_uci: str = None):
        self.state = state        # may be None for placeholder children
        self.parent = parent
        self.move = move_uci
        self.prior = float(prior)
        self.children = {}  # move_uci -> MCTSNode
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.antiQ = 0.0
        self.variance = 1.0
        self.antivariance = 1.0
        self.virtual_loss = 0
        self.is_expanded = False

    def is_root(self):
        return self.parent is None

    def expand(self, policy: dict):
        """
        Create lightweight child placeholders only. Do not materialize full Node objects here.
        """
        # prepare placeholders outside lock (fast)
        placeholders = {}
        for mv, p in policy.items():
            placeholders[mv] = MCTSNode(state=None, parent=self, prior=float(p), move_uci=mv)

        # attach children under lock (minimize contention)
        # caller typically already holds a lock, but be defensive
        self.children.update(placeholders)
        self.is_expanded = True

    def total_children_visits(self) -> int:
        return sum(ch.N for ch in self.children.values())

    def update_Q(self):
        # recompute stats from children (simple average-weighted version)
        # keep it lightweight
        self.N = sum(ch.N for ch in self.children.values())
        if self.N == 0:
            return
        # W is accumulated wins already; recompute Q if you track differently
        self.Q = self.W / max(1, self.N)
# ...existing code...

    def materialize_state(self):
        """
        If this node is a placeholder (state is None), build the Node by applying
        its move to parent.state.board. This should be called just before evaluation.
        """
        if self.state is not None:
            return self.state
        if self.parent is None or self.parent.state is None:
            raise RuntimeError("Cannot materialize root-less or parent-less node")
        # make a fast shallow copy of board and apply move
        board_copy = self.parent.state.board.copy(stack=False)
        board_copy.push(chess.Move.from_uci(self.move))
        # copy parent's context pointer (cheap) — evaluator may update it later
        parent_ctx = self.parent.state.context if hasattr(self.parent.state, "context") else None
        self.state = Node(board_copy, move=None, context=(parent_ctx.clone() if parent_ctx is not None else None))
        return self.state

# ---- MCTS engine ----
class MCTS:
    def __init__(self, evaluator=None, c_puct: float = 1.2, dirichlet_alpha: float = None, epsilon: float = 0.25,
                 batch_size: int = 8, num_workers: int = 4, virtual_loss: int = 1, profile: bool = False):
        """
        evaluator: callable(node: Node) -> (policy_dict, value_float, updated_context_tensor)
                   - policy_dict: {move_uci: prob}
                   - value_float: in [-1,1] from perspective of node.state.board.turn
        c_puct: exploration constant
        dirichlet_alpha/epsilon: optional root noise parameters
        profile: if True, collect timing stats and return them from search()
        """
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.epsilon = epsilon
        # batching / parallel eval parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        # how much virtual loss to apply per visit during selection
        self.virtual_loss = virtual_loss
        # simple cache for evaluated positions (keyed by FEN)
        self.eval_cache = {}
        # lock for thread-safe tree updates
        self.lock = threading.Lock()
        # persistent pools (created once)
        self._process_pool = None
        self._thread_pool = ThreadPoolExecutor(max_workers=self.num_workers)
        # profiling flag & stats
        self.profile = profile
        self.profile_stats = {
            "selection_time": 0.0,
            "eval_time": 0.0,
            "expand_time": 0.0,
            "backup_time": 0.0,
            "movegen_time": 0.0,
            "eval_count": 0,
            "batches": 0,
            "sims": 0
        }
        # try to create persistent process pool if an evaluator is provided and picklable
        if self.evaluator is not None:
            try:
                eval_bytes = pickle.dumps(self.evaluator)
                self._process_pool = ProcessPoolExecutor(max_workers=self.num_workers,
                                                         initializer=_worker_init,
                                                         initargs=(eval_bytes,))
            except Exception:
                self._process_pool = None

    def _default_evaluate(self, node: Node):
        legal = list(node.board.legal_moves)
        pol = {m.uci(): 1.0 / max(1, len(legal)) for m in legal}
        value = 0.0
        antivalue = 0.0
        variance = 1.0  # Default uncertainty for value
        antivariance = 1.0  # Default uncertainty for antivalue
        return pol, value, antivalue, variance, antivariance, node.context

    def _select(self, root: MCTSNode):
        t0 = time.perf_counter() if self.profile else None
        path = [root]
        node = root
        root_turn = root.state.board.turn
        while node.is_expanded and node.children:
            parent_N = max(1, node.N)
            best_score = -float('inf')
            best_child = None
            # Keep children in a stable list order
            children = list(node.children.values())

            # Build value list for selection from the root's perspective.
            # Swap Q/antiQ for opponent nodes and incorporate variance/antivariance
            vals = []
            vars_for_weights = []
            for ch in children:
                if ch.state is None:
                    ch.materialize_state()
                same_turn = (ch.state.board.turn == root_turn)
                val = ch.Q if same_turn else ch.antiQ
                var = ch.variance if same_turn else ch.antivariance
                vals.append(val)
                # avoid divide-by-zero for softmax input scaling
                vars_for_weights.append(max(var, 1e-9))

            vals_tensor = torch.tensor(vals, dtype=torch.float32)
            inv_var = torch.tensor([1.0 / v for v in vars_for_weights], dtype=torch.float32)
            weight_inputs = vals_tensor * inv_var
            q_weights = softmax(weight_inputs)

            for i, ch in enumerate(children):
                same_turn = (ch.state.board.turn == root_turn)
                # exploration scaled by the variance appropriate to the child
                var_for_U = ch.variance if same_turn else ch.antivariance
                # include virtual loss in denominator to penalize busy children
                denom = (1 + ch.N + ch.virtual_loss)
                U = self.c_puct * ch.prior * math.sqrt(parent_N) / denom * max(var_for_U, 1e-9)
                q_val = ch.Q if same_turn else ch.antiQ
                Q = q_val * float(q_weights[i])
                score = Q + U
                if score > best_score:
                    best_score = score
                    best_child = ch
            node = best_child
            # mark virtual loss while this path is being explored (discourages other sims)
            if node is not None:
                node.virtual_loss += self.virtual_loss
            path.append(node)
            if node is None or node.state.board.is_game_over():
                break
        if self.profile:
            self.profile_stats["selection_time"] += (time.perf_counter() - t0)
        return node, path

    def _backup(self, path, value: float, antivalue: float):
        bt0 = time.perf_counter() if self.profile else None
        # Propagate value and antivalue up the path
        for i, node in enumerate(reversed(path)):
            if node is None:
                continue
            if i == 0:
                # Leaf node: set W/antiW directly and mark certainty at leaf
                node.W = value
                node.antiW = antivalue
                node.N = 1
                node.variance = 0.0
                node.antivariance = 0.0
            else:
                node.update_Q()
        # remove virtual loss markers along the path
        for node in path:
            if node is None:
                continue
            node.virtual_loss = max(0, node.virtual_loss - self.virtual_loss)
        if self.profile:
            self.profile_stats["backup_time"] += (time.perf_counter() - bt0)

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
                variance = 0.0
                antivariance = 0.0
                policy = {}
                return policy, value, antivalue, variance, antivariance, mnode.state.context
            value = 1.0 if winner == board.turn else -1.0
            antivalue = -value
            variance = 0.0
            antivariance = 0.0
            return {}, value, antivalue, variance, antivariance, mnode.state.context

        # use the dedicated evaluator accessor which can be cached / batched
        et0 = time.perf_counter() if self.profile else None
        policy, value, antivalue, variance, antivariance, new_ctx = self._eval_node(mnode)
        if self.profile:
            self.profile_stats["eval_time"] += (time.perf_counter() - et0)

        ex0 = time.perf_counter() if self.profile else None
        mnode.expand(policy)
        if self.profile:
            self.profile_stats["expand_time"] += (time.perf_counter() - ex0)

        mnode.Q = float(value)
        mnode.antiQ = float(antivalue)
        mnode.variance = float(variance)
        mnode.antivariance = float(antivariance)
        mnode.is_expanded = True
        return policy, float(value), float(antivalue), float(variance), float(antivariance), new_ctx

    def _eval_node(self, mnode: MCTSNode):
        """
        Evaluate a node without mutating the tree. Results are cached by FEN.
        Returns: policy, value, antivalue, variance, antivariance, ctx
        """
        t0 = time.perf_counter() if self.profile else None
        board = mnode.state.board
        fen = board.fen()
        # cache check
        cached = self.eval_cache.get(fen)
        if cached is not None:
            if self.profile:
                self.profile_stats["eval_count"] += 1
            return cached

        # terminal
        if board.is_game_over():
            res = board.result()
            if res == '1-0':
                winner = chess.WHITE
            elif res == '0-1':
                winner = chess.BLACK
            else:
                policy = {}
                value = 0.0
                antivalue = 0.0
                variance = 0.0
                antivariance = 0.0
                self.eval_cache[fen] = (policy, value, antivalue, variance, antivariance, mnode.state.context)
                if self.profile:
                    self.profile_stats["eval_count"] += 1
                return policy, value, antivalue, variance, antivariance, mnode.state.context
            value = 1.0 if winner == board.turn else -1.0
            antivalue = -value
            variance = 0.0
            antivariance = 0.0
            self.eval_cache[fen] = ({}, value, antivalue, variance, antivariance, mnode.state.context)
            if self.profile:
                self.profile_stats["eval_count"] += 1
            return {}, value, antivalue, variance, antivariance, mnode.state.context

        # non-terminal: call provided evaluator or default
        if self.evaluator is None:
            # measure move generation time for default evaluator
            mg0 = time.perf_counter() if self.profile else None
            policy, value, antivalue, variance, antivariance, ctx = self._default_evaluate(mnode.state)
            if self.profile:
                self.profile_stats["movegen_time"] += (time.perf_counter() - mg0)
                self.profile_stats["eval_count"] += 1
        else:
            try:
                mg0 = time.perf_counter() if self.profile else None
                res = self.evaluator(mnode.state)
                if self.profile:
                    # many evaluators internally call legal_moves -> approximate as movegen time here
                    self.profile_stats["movegen_time"] += (time.perf_counter() - mg0)
            except Exception:
                res = None
            if res is None:
                policy, value, antivalue, variance, antivariance, ctx = self._default_evaluate(mnode.state)
            else:
                # expected format: value, antivalue, raw_policy, variance, antivariance, ctx
                try:
                    value, antivalue, raw_policy, variance, antivariance, ctx = res
                except Exception:
                    # fallback: evaluator might return policy first
                    if isinstance(res, (tuple, list)) and len(res) >= 1 and isinstance(res[0], dict):
                        # assume (policy_dict, value, ...)
                        pd = res[0]
                        val = float(res[1]) if len(res) > 1 else 0.0
                        aval = float(res[2]) if len(res) > 2 else -val
                        policy = {k: float(v) for k, v in pd.items()}
                        self.eval_cache[fen] = (policy, val, aval, 1.0, 1.0, mnode.state.context)
                        if self.profile:
                            self.profile_stats["eval_count"] += 1
                        return policy, val, aval, 1.0, 1.0, mnode.state.context
                    # otherwise fallback to default
                    policy, value, antivalue, variance, antivariance, ctx = self._default_evaluate(mnode.state)

                # construct policy from raw_policy vector or dict
                policy = {}
                mg0 = time.perf_counter() if self.profile else None
                legal_moves = list(board.legal_moves)
                if isinstance(raw_policy, dict):
                    legal_uci = {m.uci() for m in legal_moves}
                    policy = {k: float(v) for k, v in raw_policy.items() if k in legal_uci}
                else:
                    import torch as _torch
                    logits = _torch.tensor([
                        raw_policy[moves.pmove_to_idx.get(m.uci(), -1)] if m.uci() in moves.pmove_to_idx else float('-inf')
                        for m in legal_moves
                    ], dtype=_torch.float32)
                    mask = _torch.tensor([m.uci() in moves.pmove_to_idx for m in legal_moves], dtype=_torch.bool)
                    if logits.numel() > 0:
                        logits[~mask] = float('-inf')
                        probs = _torch.softmax(logits, dim=0)
                        for i, m in enumerate(legal_moves):
                            if mask[i]:
                                policy[m.uci()] = float(probs[i])
                if not policy and legal_moves:
                    policy = {m.uci(): 1.0 / len(legal_moves) for m in legal_moves}
                if self.profile:
                    self.profile_stats["movegen_time"] += (time.perf_counter() - mg0)
                    self.profile_stats["eval_count"] += 1

        # normalize
        s = float(sum(policy.values())) if policy else 0.0
        if s > 0:
            for k in list(policy.keys()):
                policy[k] = float(policy[k]) / s

        self.eval_cache[fen] = (policy, float(value), float(antivalue), float(variance), float(antivariance), ctx)
        if self.profile:
            self.profile_stats["eval_time"] += (time.perf_counter() - t0)
        return self.eval_cache[fen]

    def search(self, root_state: Node, num_sims: int = 100, profile: bool = False):
        # if caller requests profiling explicitly override instance flag for this run
        run_profile = self.profile or profile
        if run_profile and not self.profile:
            self.profile = True
        root = MCTSNode(root_state, parent=None, prior=1.0)

        # Obtain evaluator output (support several possible formats)
        if self.evaluator:
            res = self.evaluator(root_state)
        else:
            res = self._default_evaluate(root_state)

        policy = {}
        try:
            if res is None:
                policy, _, _, _, _, _ = self._default_evaluate(root_state)
            else:
                # Prefer the (value, antivalue, raw_policy, variance, antivariance, ctx) format
                if isinstance(res, (tuple, list)) and len(res) == 6:
                    value, antivalue, raw_policy, variance, antivariance, _ctx = res
                    board = root_state.board
                    mg0 = time.perf_counter() if run_profile else None
                    legal_moves = list(board.legal_moves)
                    if run_profile:
                        self.profile_stats["movegen_time"] += (time.perf_counter() - mg0)
                    if isinstance(raw_policy, dict):
                        # use provided dict but restrict to legal moves
                        legal_uci = {m.uci() for m in legal_moves}
                        policy = {k: float(v) for k, v in raw_policy.items() if k in legal_uci}
                    else:
                        # raw_policy is expected to be a list/tensor indexed by pmove_to_idx
                        import torch
                        logits = torch.tensor(
                            [raw_policy[moves.pmove_to_idx.get(m.uci(), -1)] if m.uci() in moves.pmove_to_idx else float('-inf')
                             for m in legal_moves],
                            dtype=torch.float32
                        )
                        mask = torch.tensor([m.uci() in moves.pmove_to_idx for m in legal_moves], dtype=torch.bool)
                        logits[~mask] = float('-inf')
                        if logits.numel() > 0:
                            probs = torch.softmax(logits, dim=0)
                            for i, m in enumerate(legal_moves):
                                if mask[i]:
                                    policy[m.uci()] = float(probs[i])
                        if not policy and legal_moves:
                            policy = {m.uci(): 1.0 / len(legal_moves) for m in legal_moves}
                else:
                    # Try to interpret first element as a policy dict or list
                    if isinstance(res[0], dict):
                        policy = {k: float(v) for k, v in res[0].items()}
                    elif isinstance(res[0], (list, tuple)):
                        raw_policy = res[0]
                        board = root_state.board
                        mg0 = time.perf_counter() if run_profile else None
                        legal_moves = list(board.legal_moves)
                        if run_profile:
                            self.profile_stats["movegen_time"] += (time.perf_counter() - mg0)
                        import torch
                        logits = torch.tensor(
                            [raw_policy[moves.pmove_to_idx.get(m.uci(), -1)] if m.uci() in moves.pmove_to_idx else float('-inf')
                             for m in legal_moves],
                            dtype=torch.float32
                        )
                        mask = torch.tensor([m.uci() in moves.pmove_to_idx for m in legal_moves], dtype=torch.bool)
                        logits[~mask] = float('-inf')
                        if logits.numel() > 0:
                            probs = torch.softmax(logits, dim=0)
                            for i, m in enumerate(legal_moves):
                                if mask[i]:
                                    policy[m.uci()] = float(probs[i])
                        if not policy and legal_moves:
                            policy = {m.uci(): 1.0 / len(legal_moves) for m in legal_moves}
        except Exception:
            policy = {}

        # Normalize or fall back to uniform over legal moves
        if policy:
            s = float(sum(policy.values()))
            if s <= 0:
                policy = {}
            else:
                for k in list(policy.keys()):
                    policy[k] = float(policy[k]) / s

        if not policy:
            legal = list(root_state.board.legal_moves)
            policy = {m.uci(): 1.0 / max(1, len(legal)) for m in legal}

        # Add Dirichlet noise to root if configured
        if self.dirichlet_alpha is not None:
            moves_list = list(policy.keys())
            if moves_list:
                noise = [random.gammavariate(self.dirichlet_alpha, 1.0) for _ in moves_list]
                s = sum(noise) or 1.0
                noise = [n / s for n in noise]
                for i, m in enumerate(moves_list):
                    policy[m] = policy.get(m, 0.0) * (1 - self.epsilon) + noise[i] * self.epsilon

        root.expand(policy)
        root.is_expanded = True

        sims_done = 0
        pbar = tqdm.tqdm(total=num_sims, desc="MCTS Simulations")
        while sims_done < num_sims:
            # Collect a batch of leaves/paths via selection
            batch_leaves = []
            batch_paths = []
            to_collect = min(self.batch_size, num_sims - sims_done)
            sel_t0 = time.perf_counter() if run_profile else None
            for _ in range(to_collect):
                leaf, path = self._select(root)
                batch_leaves.append(leaf)
                batch_paths.append(path)
            if run_profile:
                self.profile_stats["selection_time"] += (time.perf_counter() - sel_t0)
            # De-duplicate leaves for evaluation
            unique_leaves = []
            seen = set()
            for leaf in batch_leaves:
                if leaf is None:
                    continue
                lid = id(leaf)
                if lid in seen:
                    continue
                seen.add(lid)
                unique_leaves.append(leaf)

            # Evaluate unique leaves in parallel (non-mutating). Prefer persistent multiprocessing pool when available.
            results = {}
            if unique_leaves:
                self.profile_stats["batches"] += 1
                # prepare tasks as (fen, board_tensor_np, ctx_np)
                tasks = []
                for leaf in unique_leaves:
                    try:
                        bt_np = leaf.state.board_tensor.cpu().numpy()
                        ctx_np = leaf.state.context.cpu().numpy()
                    except Exception:
                        # fallback to converting via torch
                        bt_np = leaf.state.board_tensor.numpy()
                        ctx_np = leaf.state.context.numpy()
                    tasks.append((leaf, (leaf.state.board.fen(), bt_np, ctx_np)))

                # Try multiprocessing first; fall back to threads on failure.
                used_mp = False
                # try persistent process pool first
                if self._process_pool is not None:
                    try:
                        eval_t0 = time.perf_counter() if run_profile else None
                        fut_map = {self._process_pool.submit(_worker_eval, t[1]): t[0] for t in tasks}
                        for fut in fut_map:
                            leaf = fut_map[fut]
                            try:
                                fen, out = fut.result()
                                if out is None:
                                    results[leaf] = self._default_evaluate(leaf.state)
                                else:
                                    results[leaf] = out
                            except Exception:
                                results[leaf] = self._default_evaluate(leaf.state)
                        if run_profile:
                            self.profile_stats["eval_time"] += (time.perf_counter() - eval_t0)
                        used_mp = True
                    except Exception:
                        used_mp = False

                if not used_mp:
                    # threaded fallback using persistent thread pool
                    eval_t0 = time.perf_counter() if run_profile else None
                    futs = {self._thread_pool.submit(self._eval_node, leaf): leaf for leaf in unique_leaves}
                    for fut in futs:
                        leaf = futs[fut]
                        try:
                            results[leaf] = fut.result()
                        except Exception:
                            results[leaf] = self._default_evaluate(leaf.state)
                    if run_profile:
                        self.profile_stats["eval_time"] += (time.perf_counter() - eval_t0)

            # Expand and update nodes under lock to avoid races
            with self.lock:
                ex_t0 = time.perf_counter() if run_profile else None
                for leaf in unique_leaves:
                    policy, value, antivalue, variance, antivariance, new_ctx = results.get(leaf, self._default_evaluate(leaf.state))
                    if not leaf.is_expanded:
                        leaf.expand(policy)
                        leaf.Q = float(value)
                        leaf.antiQ = float(antivalue)
                        leaf.variance = float(variance)
                        leaf.antivariance = float(antivariance)
                        leaf.is_expanded = True
                if run_profile:
                    self.profile_stats["expand_time"] += (time.perf_counter() - ex_t0)

            # Backup per selected path using evaluated values
            for leaf, path in zip(batch_leaves, batch_paths):
                if leaf is None:
                    sims_done += 1
                    pbar.update(1)
                    continue
                policy, value, antivalue, variance, antivariance, new_ctx = results.get(leaf, self._default_evaluate(leaf.state))
                self._backup(path, value, antivalue)
                sims_done += 1
                self.profile_stats["sims"] += 1
                pbar.update(1)
        pbar.close()

        distr = {child.move: child.N for child in root.children.values()}
        best_child = max(root.children.values(), key=lambda ch: ch.N, default=None)
        best_move = best_child.move if best_child else None
        if run_profile:
            # return stats as 4th return value to avoid breaking callers using profile=False
            return best_child, best_move, distr, dict(self.profile_stats)
        return best_child, best_move, distr

    def close(self):
        """Shut down persistent pools."""
        try:
            if self._process_pool is not None:
                self._process_pool.shutdown(wait=True)
                self._process_pool = None
        except Exception:
            pass
        try:
            if self._thread_pool is not None:
                self._thread_pool.shutdown(wait=True)
                self._thread_pool = None
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


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

