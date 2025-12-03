import math
import weakref
import random
import torch
import chess
import moves  # assuming you have this with pmove_to_idx
import tqdm
import threading
import pickle
import time
import multiprocessing
from typing import Dict, Tuple, Optional, List, Any
import lmdb
import os
import numpy as _np
TEMPERATURE = 250
DEPTH_NOISE_END = 10
DEPTH_NOISE_SCALE = 100.0

# open a writable LMDB for main process; workers will open their own readonly handle
LMDB_PATH = './lmdb_data'
os.makedirs(LMDB_PATH, exist_ok=True)
env = lmdb.open(LMDB_PATH, map_size=2**30, max_dbs=0)

CONTEXT_LENGTH = 256*4
MAX_PIECES = 33
def _unpack_ctx(ctx_obj):
    """
    Safely unpack a context which may be:
      - None
      - pickled bytes / bytearray
      - a torch.Tensor
      - already a numpy/other object
    Returns the raw context object or None.
    """
    if ctx_obj is None:
        return None
    if isinstance(ctx_obj, (bytes, bytearray)):
        try:
            return pickle.loads(ctx_obj)
        except Exception:
            return None
    if isinstance(ctx_obj, torch.Tensor):
        return ctx_obj
    try:
        # best-effort: try to unpickle, otherwise return as-is
        return pickle.loads(ctx_obj)
    except Exception:
        return ctx_obj
# helper LMDB helpers
def lmdb_get_node(node_id: int, writable: bool = False) -> Optional[Dict[str, Any]]:
    path = f'node_{node_id}'.encode()
    # child processes should open their own env; main process can reuse global env
    if writable:
        with env.begin(write=True) as txn:
            data = txn.get(path)
    else:
        # readonly access (safe for workers if they open their own env in their process)
        with env.begin(write=False) as txn:
            data = txn.get(path)
    return pickle.loads(data) if data is not None else None

def lmdb_put_node(node_dict: Dict[str, Any]):
    key = f"node_{node_dict['id']}".encode()
    with env.begin(write=True) as txn:
        txn.put(key, pickle.dumps(node_dict))

# ========================================
# Node: lightweight board + tensor encoding
# ========================================
class Node:
    def __init__(self, board: chess.Board, context: Optional[torch.Tensor] = None):
        self.board = board
        self.board_tensor = self._to_tensor()
        # robustly handle context whether it's torch.Tensor, numpy or None
        if context is None:
            self.context = torch.zeros(CONTEXT_LENGTH, dtype=torch.float32)
        else:
            if isinstance(context, torch.Tensor):
                self.context = context.clone().detach().float()
            else:
                arr = _np.asarray(context, dtype=_np.float32).flatten()
                if arr.size < CONTEXT_LENGTH:
                    tmp = _np.zeros(CONTEXT_LENGTH, dtype=_np.float32)
                    tmp[:arr.size] = arr
                    arr = tmp
                else:
                    arr = arr[:CONTEXT_LENGTH]
                self.context = torch.from_numpy(arr).float()

    def _to_tensor(self) -> torch.Tensor:
        tens = torch.zeros((MAX_PIECES, 4), dtype=torch.float32)
        i = 0
        for square, piece in self.board.piece_map().items():
            if i >= MAX_PIECES - 1:
                break
            tens[i, 0] = piece.piece_type - 1
            tens[i, 1] = float(int(piece.color))
            tens[i, 2] = (square // 8) - 3.5
            tens[i, 3] = (square % 8) - 3.5
            i += 1
        # sentinel row: keep ply in a single slot instead of a 3-element slice
        tens[MAX_PIECES-1, 0] = 10.0
        tens[MAX_PIECES-1, 1] = float(self.board.ply())
        return tens

    def to(self, device):
        self.board_tensor = self.board_tensor.to(device)
        self.context = self.context.to(device)
        return self
global id
id=0

# ========================================
# MCTS Node with your beloved variance madness
# ========================================
class MCTSNode:
    def __init__(self, state: Node, parent: Optional['MCTSNode'], prior: float, move_uci: Optional[str], turn: chess.Color):
        self.state = state
        self.parent = weakref.ref(parent)
        self.move = move_uci
        self.turn = turn
        self.prior = float(prior)
        global id
        self.id=id
        id+=1
        self.children: Dict[str, MCTSNode] = {}
        self.N = 0
        self.W = 0.0        # sum of values (from root's perspective)
        self.W_anti = 0.0   # sum of antivalues
        self.Q = 0.0
        self.antiQ = 0.0
        self.variance = 1.0

        self.virtual_loss = 0
        self.is_expanded = False

        self.opponent_softmax_weights: Optional[torch.Tensor] = None  # cached for selection

        # persist minimal serializable representation to LMDB
        lmdb_put_node(self.to_dict())
    def __del__(self):
        with env.begin(write=True) as txn:
            key = f'node_{self.id}'.encode()
            txn.delete(key)
    def recompute_stats_and_weights(self, root_turn: chess.Color):
        if self.N > 0:
            self.Q = self.W / self.N
            self.antiQ = self.W_anti / self.N

        if not self.children:
            self.opponent_softmax_weights = None
            return

        children = list(self.children.values())
        opponent_vals = []
        vars=[]
        for ch in children:
            if ch.turn == root_turn:  # opponent to move at child
                opponent_vals.append(ch.antiQ)
            else:
                opponent_vals.append(ch.Q)
            vars.append(ch.variance)
        vals = torch.tensor(opponent_vals, dtype=torch.float32)
        vars_t = torch.tensor(vars, dtype=torch.float32)
        weights = torch.softmax(vals/TEMPERATURE, dim=0)
        self.opponent_softmax_weights = weights

    def to_dict(self):
        # store a compact dict used by workers/processes
        try:
            ctx_pickled = pickle.dumps(self.state.context)
        except Exception:
            ctx_pickled = None
        return {
            'id': self.id,
            'state': self.state.board.fen(),
            'move': self.move,
            'N': self.N,
            'W': self.W,
            'W_anti': self.W_anti,
            'Q': self.Q,
            'antiQ': self.antiQ,
            'Q_weights': self.opponent_softmax_weights.tolist() if self.opponent_softmax_weights is not None else None,
            'variance': self.variance,
            'is_expanded': self.is_expanded,
            'children': {uci: child.id for uci, child in self.children.items()},
            'turn': int(bool(self.turn)),
            'prior': float(self.prior),
            'virtual_loss': int(self.virtual_loss),
            'context': ctx_pickled
        }


# remove incomplete helpers and replace with working worker and mp orchestration
def worker_process(root_id: int, env_path: str, eval_request_queue: multiprocessing.Queue, shutdown_event: multiprocessing.Event,c_puct: float = 2.5,dechirlet_alpha: float = 0.3,  dechirlet_epsilon: float = 0.25):
    """
    Worker runs selection only, reading the shared tree from LMDB (readonly).
    When a leaf is found it places a request on eval_request_queue:
      (leaf_id, path_ids)
    The main process will perform evaluation, expansion and backup.
    """
    # open readonly env in this process
    wenv = lmdb.open(env_path, readonly=True, lock=False)
    def get_node(txn, nid):
        b = txn.get(f'node_{nid}'.encode())
        return pickle.loads(b) if b is not None else None

    selection_count = 0
    while not shutdown_event.is_set():
        try:
            with wenv.begin() as txn:
                root_b = txn.get(f'node_{root_id}'.encode())
                if root_b is None:
                    time.sleep(0.01)
                    continue
                node = pickle.loads(root_b)
                path = [node['id']]
                root_turn = node.get('turn', 1)
                # traverse until leaf
                while node.get('is_expanded') and node.get('children'):
                    children_ids = list(node['children'].values())
                    child_dicts = []
                    for cid in children_ids:
                        cd = get_node(txn, cid)
                        if cd is not None:
                            child_dicts.append(cd)
                    if not child_dicts:
                        break
                    opponent_vals = []
                    priors=[]
                    for ch in child_dicts:
                        priors.append(ch.get('prior', 0.0))
                    Ns = []
                    opponent_vals = node.get('Q_weights', None)
                    weights = torch.tensor(opponent_vals, dtype=torch.float32) if opponent_vals is not None else torch.ones(len(child_dicts), dtype=torch.float32)
                    
                    # compute depth-based noise factor and use it to jitter scores
                    depth = max(0, len(path) - 1)
                    depth_factor = max(0, depth - DEPTH_NOISE_END)
                    # traverse children and pick using softmax-exploitation + U (+ noise)
                    parent_N = max(1, sum(max(1, ch.get('N', 0)) for ch in child_dicts))
                    best_score = -float('inf')
                    best_idx = 0
                    for i, ch in enumerate(child_dicts):
                        exploit_val = ch.get('Q', 0.0) if ch.get('turn', 1) != root_turn else ch.get('antiQ', 0.0)
                        exploitation = float(weights[i]) * float(exploit_val)
                        variance_term = 1.0 + ch.get('variance', 1.0)
                        u = c_puct * float(priors[i]) * math.sqrt(parent_N) / (1 + ch.get('N', 0)) * variance_term
                        score = exploitation + u
                        # add Gaussian jitter that increases with depth to reduce determinism deeper in the tree
                        if depth_factor > 0:
                            noise_std = depth_factor * DEPTH_NOISE_SCALE
                            score += random.gammavariate(dechirlet_alpha,0.1)*dechirlet_epsilon*noise_std
                        if score > best_score:
                            best_score = score
                            best_idx = i
                
                    node = child_dicts[best_idx]
                    path.append(node['id'])
                # leaf found -> request evaluation
                leaf_id = node['id']
                # send state fen and pickled context
                eval_request_queue.put((leaf_id, node['state'], node.get('context'), path))
        except Exception as e:
            # avoid noisy trace in worker; short backoff
            time.sleep(0.01)
            print("Worker exception:", e)
            continue

# ========================================
# The Beast: Optimized MCTS
# ========================================
class MCTS:
    def __init__(self, evaluator=None, c_puct: float = 2.5, dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25, batch_size: int = 16, num_workers: int = 4):
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.batch_size = batch_size
        self.num_workers = max(1, num_workers)

        self.eval_cache: Dict[str, Tuple] = {}
        self.lock = multiprocessing.Lock()  # used when main process updates LMDB stats/backup

        # worker control
        self._workers: List[multiprocessing.Process] = []
        self._eval_request_queue: Optional[multiprocessing.Queue] = None
        self._shutdown_event: Optional[multiprocessing.Event] = None

    def _default_eval(self, node: Node) -> Tuple[dict, float, float, float, torch.Tensor]:
        legal = list(node.board.legal_moves)
        n = max(1, len(legal))
        policy = {m.uci(): 1.0 / n for m in legal}
        return policy, 0.0, 0.0, 1.0, node.context
    def _eval_node(self, mnode: MCTSNode) -> Tuple[dict, float, float, float, torch.Tensor]:


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
            
            return policy, v, -v, 0.0, 0.0, mnode.state.context

        if self.evaluator is None:
            res = self._default_eval(mnode.state)
        else:
            try:
                raw = self.evaluator(mnode.state)
                if raw is None or not isinstance(raw, (tuple, list)):
                    res = self._default_eval(mnode.state)
                else:
                    # Support evaluator returning either 5 or 6 items:
                    # (v, av, pol_raw, var, ctx) OR (v, av, pol_raw, var, avar, ctx)
                    
                    if len(raw) == 5:
                        v, av, pol_raw, var, ctx = raw
                        avar = var
                    else:
                        # unexpected shape -> fallback
                        res = self._default_eval(mnode.state)
                        return res

                # build legal policy
                legal = list(board.legal_moves)
                policy = {}
                if isinstance(pol_raw, dict):
                    legal_set = {m.uci() for m in legal}
                    policy = {k: float(vv) for k, vv in pol_raw.items() if k in legal_set}
                else:
                    logits = torch.full((len(legal),), float('-inf'))
                    for i, m in enumerate(legal):
                        idx = moves.pmove_to_idx.get(m.uci(), -1)
                        if 0 <= idx < len(pol_raw):
                            logits[i] = float(pol_raw[idx])
                    if logits.isfinite().any():
                        probs = torch.softmax(logits / max(1e-6, TEMPERATURE), dim=0)
                        for m, p in zip(legal, probs.tolist()):
                            policy[m.uci()] = p
                if not policy:
                    policy = {m.uci(): 1.0/len(legal) for m in legal}
                res = (policy, float(v), float(av), float(var), float(avar), ctx)
            except Exception:
                res = self._default_eval(mnode.state)

        return res
    def _batch_default_eval(self, nodes: List[Node]) -> List[Tuple[dict, float, float, float, Any]]:
        results = []
        for node in nodes:
            legal = list(node.board.legal_moves)
            n = max(1, len(legal))
            policy = {m.uci(): 1.0 / n for m in legal}
            # return pickled context so callers can pass it directly into _expand_and_backup
            ctx_pickled = pickle.dumps(node.context) if node.context is not None else None
            results.append((policy, 0.0, 0.0, 1.0, ctx_pickled))
        return results

    def _batch_eval_nodes(self, mnodes: List[Any]) -> List[Tuple[dict, float, float, float, Any]]:
        """
        Evaluate a batch of nodes. Accepts either MCTSNode instances or Node instances.
        Returns a list of 5-tuples: (policy_dict, v, av, var, ctx_pickled)
        Tries to call evaluator in batch when possible; falls back to per-node evaluation.
        """
        # normalize to Node objects
        nodes: List[Node] = []
        for m in mnodes:
            if isinstance(m, MCTSNode):
                nodes.append(m.state)
            else:
                nodes.append(m)

        # fast path: no evaluator -> default batch
        if self.evaluator is None:
            return self._batch_default_eval(nodes)

        results: List[Tuple[dict, float, float, float, Any]] = []

        # try batch evaluator (if evaluator supports list input)
        try:
            batch_raw = self.evaluator(nodes)
            if isinstance(batch_raw, (list, tuple)) and len(batch_raw) == len(nodes):
                for node, raw in zip(nodes, batch_raw):
                    if raw is None or not isinstance(raw, (tuple, list)):
                        results.append(( {m.uci(): 1.0/max(1,len(list(node.board.legal_moves))) for m in node.board.legal_moves},
                                         0.0, 0.0, 1.0,
                                         pickle.dumps(node.context) if node.context is not None else None))
                        continue
                    # support 5- or 6-element returns

                    if len(raw) == 5:
                        v, av, pol_raw, var, ctx = raw
                    else:
                        results.append(( {m.uci(): 1.0/max(1,len(list(node.board.legal_moves))) for m in node.board.legal_moves},
                                         0.0, 0.0, 1.0,
                                         pickle.dumps(node.context) if node.context is not None else None))
                        continue

                    # normalize policy to dict over legal moves
                    if isinstance(pol_raw, dict):
                        legal_set = {m.uci() for m in node.board.legal_moves}
                        policy = {k: float(vv) for k, vv in pol_raw.items() if k in legal_set}
                    else:
                        legal = list(node.board.legal_moves)
                        policy = {}
                        if legal:
                            logits = torch.full((len(legal),), float('-inf'))
                            for i, m in enumerate(legal):
                                idx = moves.pmove_to_idx.get(m.uci(), -1)
                                if 0 <= idx < len(pol_raw):
                                    logits[i] = float(pol_raw[idx])
                            if logits.isfinite().any():
                                probs = torch.softmax(logits / max(1e-6, TEMPERATURE), dim=0)
                                policy = {m.uci(): p for m, p in zip(legal, probs.tolist())}
                            else:
                                policy = {m.uci(): 1.0 / len(legal) for m in legal}
                    ctx_pickled = pickle.dumps(ctx) if not isinstance(ctx, (bytes, bytearray)) else ctx
                    results.append((policy, float(v), float(av), float(var), ctx_pickled))
                return results
        except Exception:
            # fall through to per-node evaluation on any batch-eval failure
            pass

        # fallback: evaluate nodes one-by-one
        for node in nodes:
            try:
                raw = self.evaluator(node)
                if raw is None or not isinstance(raw, (tuple, list)):
                    policy, v, av, var, ctx_pickled = self._default_eval(node)
                    ctx_pickled = pickle.dumps(ctx_pickled) if not isinstance(ctx_pickled, (bytes, bytearray)) else ctx_pickled
                    results.append((policy, float(v), float(av), float(var), ctx_pickled))
                    continue
                if len(raw) == 5:
                    v, av, pol_raw, var, ctx = raw
                else:
                    policy, v, av, var, ctx_p = self._default_eval(node)
                    ctx_pickled = pickle.dumps(ctx_p) if not isinstance(ctx_p, (bytes, bytearray)) else ctx_p
                    results.append((policy, float(v), float(av), float(var), ctx_pickled))
                    continue

                if isinstance(pol_raw, dict):
                    legal_set = {m.uci() for m in node.board.legal_moves}
                    policy = {k: float(vv) for k, vv in pol_raw.items() if k in legal_set}
                else:
                    legal = list(node.board.legal_moves)
                    policy = {}
                    if legal:
                        logits = torch.full((len(legal),), float('-inf'))
                        for i, m in enumerate(legal):
                            idx = moves.pmove_to_idx.get(m.uci(), -1)
                            if 0 <= idx < len(pol_raw):
                                logits[i] = float(pol_raw[idx])
                        if logits.isfinite().any():
                            probs = torch.softmax(logits / max(1e-6, TEMPERATURE), dim=0)
                            policy = {m.uci(): p for m, p in zip(legal, probs.tolist())}
                        else:
                            policy = {m.uci(): 1.0 / len(legal) for m in legal}
                ctx_pickled = pickle.dumps(ctx) if not isinstance(ctx, (bytes, bytearray)) else ctx
                results.append((policy, float(v), float(av), float(var), ctx_pickled))
            except Exception:
                policy, v, av, var, ctx_p = self._default_eval(node)
                ctx_pickled = pickle.dumps(ctx_p) if not isinstance(ctx_p, (bytes, bytearray)) else ctx_p
                results.append((policy, float(v), float(av), float(var), ctx_pickled))

        return results
    def mp_search(self, node:MCTSNode, num_sims: int = 800, context: Optional[torch.Tensor] = None):
        """
        Orchestrate multiprocessing MCTS:
          - spawn worker processes to run selection from the shared LMDB tree
          - workers put leaf eval requests on a queue
          - main process batches evaluations, expands and backs up under a lock
        """
        # prepare root
        root=node

        policy, v, av, var, ctx = self._eval_node(root)
        # apply dirichlet noise to root policy if requested
        if self.dirichlet_alpha:
            moves_list = list(policy.keys())
            if moves_list:
                noise = _np.array([random.gammavariate(self.dirichlet_alpha, 1.0) for _ in moves_list], dtype=float)
                noise = noise / (noise.sum() + 1e-12)
                policy = {m: policy[m] * (1 - self.dirichlet_epsilon) + float(n) * self.dirichlet_epsilon
                          for m, n in zip(moves_list, noise)}

        # Expand root locally (persisted by MCTSNode creation)
        board_copy = root.state.board
        for uci, p in policy.items():
            move = chess.Move.from_uci(uci)
            new_board = board_copy.copy(stack=False)
            new_board.push(move)
            new_node = Node(new_board, ctx)
            child = MCTSNode(new_node, root, p, uci, not board_copy.turn)
            root.children[uci] = child
        root.is_expanded = True
        # update root on LMDB (overwrite)
        lmdb_put_node(root.to_dict())

        # create queues and workers
        eval_queue = multiprocessing.Queue(maxsize=1024)
        shutdown_event = multiprocessing.Event()
        self._eval_request_queue = eval_queue
        self._shutdown_event = shutdown_event

        # spawn worker processes
        for i in range(self.num_workers):
            p = multiprocessing.Process(target=worker_process,
                                        args=(root.id, LMDB_PATH, eval_queue, shutdown_event, self.c_puct,self.dirichlet_alpha,self.dirichlet_epsilon),
                                        daemon=True)
            p.start()
            self._workers.append(p)

        sims_done = 0
        bar = tqdm.tqdm(total=num_sims, desc="MCTS sims (MP)", colour='#00ff00')
        try:
            while sims_done < num_sims or (num_sims-sims_done < self.batch_size and sims_done != num_sims):
                batch = []
                try:
                    # collect up to batch_size requests
                    if sims_done + self.batch_size <= num_sims:

                        while len(batch) < self.batch_size:
                            item = eval_queue.get(timeout=0.2)
                            batch.append(item)
                    else:
                        # final partial batch
                        while len(batch) < (num_sims - sims_done):
                            print("sims_done:", sims_done, "num_sims:", num_sims)
                            item = eval_queue.get(timeout=0.2)
                            batch.append(item)
                
                except Exception:
                    # timeout or empty; proceed with what we have
                    pass

                if not batch:
                    continue

                # deduplicate by leaf id, keep most recent path
                unique = {}
                for leaf_id, fen, ctx_pickled, path in batch:
                    unique[leaf_id] = (fen, ctx_pickled, path)
                eval_items = list(unique.items())[:self.batch_size]  # [(leaf_id, (fen, ctx, path))]
                # prepare Node objects for evaluator
                eval_nodes = []
                leaf_ids = []
                paths = []
                for leaf_id, (fen, ctx_pickled, path) in eval_items:
                    board_local = chess.Board(fen)
                    ctx = _unpack_ctx(ctx_pickled)
                    node = Node(board_local, ctx)
                    eval_nodes.append(node)
                    leaf_ids.append(leaf_id)
                    paths.append(path)

                # run evaluator in batch (use _batch_eval_nodes)
                results = self._batch_eval_nodes(eval_nodes)

                # expand and backup each result under lock
                for i, (leaf_id, (fen, ctx_pickled, path)) in enumerate(eval_items):
                    policy, v, av, var, ctx_p = results[i]
                    self._expand_and_backup(leaf_id, policy, v, av, var, ctx_p, path)
                    sims_done += 1
                    bar.update(1)
        finally:
            bar.close()
            # shutdown workers
            shutdown_event.set()
            for p in self._workers:
                p.join(timeout=1.0)
            self._workers = []

        # after sims compute best move from root in LMDB
        root_dict = lmdb_get_node(root.id)
        if not root_dict:
            return None, {}, None
        visits = {}
        for uci, cid in root_dict.get('children', {}).items():
            child = lmdb_get_node(cid)
            visits[uci] = child.get('N', 0) if child else 0
        best_uci = max(visits, key=visits.get) if visits else None
        best_child = None
        if best_uci:
            best_child = lmdb_get_node(root_dict['children'][best_uci])
        return best_uci, visits, best_child
        # keep the legacy single-process search
    def _expand_and_backup(self, leaf_id: int, policy: dict, v: float, av: float, var: float, ctx_pickled: Any, path: List[int]):
        """
        Expand the leaf (create child nodes) and perform backup along the path.
        This function acquires a lock to serialize LMDB writes.
        """
        with self.lock:
            # reload leaf to ensure latest
            leaf = lmdb_get_node(leaf_id)
            if leaf is None:
                return
            if not leaf.get('is_expanded'):
                board = chess.Board(leaf['state'])
                # build and attach children
                for uci, p in policy.items():
                    move = chess.Move.from_uci(uci)
                    new_board = board.copy(stack=False)
                    new_board.push(move)
                    # safely unpack context (ctx_pickled may be bytes or a tensor)
                    ctx_obj = _unpack_ctx(ctx_pickled)
                    node_obj = MCTSNode(Node(new_board, ctx_obj),
                                        parent=None, prior=p, move_uci=uci, turn=not board.turn)
                    # child persisted by MCTSNode
                    # link child id in leaf dict
                    leaf_children = leaf.get('children', {})
                    leaf_children[uci] = node_obj.id
                    leaf['children'] = leaf_children
                leaf['is_expanded'] = True
                leaf['variance'] = float(var)
                # initialize leaf stats
                leaf['W'] = float(v)
                leaf['W_anti'] = float(av)
                leaf['N'] = 1
                leaf['Q'] = float(v)
                leaf['antiQ'] = float(av)
                lmdb_put_node(leaf)

            # backup along the path (path is list of ids from root..leaf)
            # we assume path contains valid node ids and exist
            for nid in reversed(path):
                nd = lmdb_get_node(nid)
                if nd is None:
                    continue
                nd['N'] = nd.get('N', 0) + 1
                nd['W'] = nd.get('W', 0.0) + v
                nd['W_anti'] = nd.get('W_anti', 0.0) + (av * 0.99 + 0.01 * nd.get('prior', 0.0))
                nd['virtual_loss'] = max(0, nd.get('virtual_loss', 0) - 0)
                # recompute Qs
                nd['Q'] = nd['W'] / max(1, nd['N'])
                nd['antiQ'] = nd['W_anti'] / max(1, nd['N'])
                nd['variance'] = float(var)
                cids= nd.get('children', {}).values()
                if cids:
                    opponent_vals = []
                    for cid in cids:
                        child = lmdb_get_node(cid)
                        if child is not None:
                            if child.get('turn', 1) == nd.get('turn', 1):
                                opponent_vals.append(child.get('antiQ', 0.0))
                            else:
                                opponent_vals.append(child.get('Q', 0.0))
                    if opponent_vals:
                        vals_t = torch.tensor(opponent_vals, dtype=torch.float32)
                        weights = torch.softmax(vals_t / TEMPERATURE, dim=0)
                        nd['Q_weights'] = weights.tolist()
                    else:
                        nd['Q_weights'] = None
                else:
                   raise ValueError("Node has no children during backup.")
                lmdb_put_node(nd)

    def search(self, board: chess.Board, num_sims: int = 800, context: Optional[torch.Tensor] = None):
        # default to mp_search if evaluator is provided
        if self.num_workers > 1 and self.evaluator is not None:
            return self.mp_search(board, num_sims=num_sims, context=context)
        # else fallback to original single-process search
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

        for _ in tqdm.tqdm(range(num_sims), desc="MCTS sims",colour='#ff0000'):
            leaf, path = self._select(root)
            if not leaf.is_expanded:
                self._expand(leaf)
                value = leaf.Q
                antivalue = leaf.antiQ
            else:
                # already expanded (rare race), just reuse
                value = leaf.Q
                antivalue = leaf.antiQ
            variance=leaf.variance
            self._backup(path, value, antivalue, variance)

        visits = {uci: child.N for uci, child in root.children.items()}
        best_uci = max(visits, key=visits.get)
        best_child = root.children[best_uci]

        return best_uci, visits, best_child

    def close(self):
        # attempt to shut down workers if any
        if self._shutdown_event:
            self._shutdown_event.set()
        for p in self._workers:
            p.join(timeout=1.0)
        self._workers = []
        # close LMDB env if needed
        try:
            env.close()
        except Exception:
            pass

# ========================================
# Quick test (uncomment to run)
# ========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from Model import  ChessAttention
def batch_evaluator(nodes: List[Node]) -> List[Tuple[dict, float, float, float, torch.Tensor]]:
    tens= torch.stack([n.board_tensor for n in nodes], dim=0).to('cpu')
    contexts= torch.stack([n.context for n in nodes], dim=0).to('cpu')
    values, antivalues,variances, policy_logits,new_contexts = model(tens, contexts)
    results = []
    for i in range(len(nodes)):
        board = nodes[i].board
        legal = list(board.legal_moves)
        policy = {}
        if legal:
            logits = torch.full((len(legal),), float('-inf'))
            for j, m in enumerate(legal):
                idx = moves.pmove_to_idx.get(m.uci(), -1)
                if 0 <= idx < policy_logits.size(1):
                    logits[j] = float(policy_logits[i, idx])
            if logits.isfinite().any():
                probs = torch.softmax(logits / max(1e-6, TEMPERATURE), dim=0)
                policy = {m.uci(): p for m, p in zip(legal, probs.tolist())}
            else:
                policy = {m.uci(): 1.0 / len(legal) for m in legal}
        results.append((policy,
                        float(values[i].item()),
                        float(antivalues[i].item()),
                        float(variances[i].item()),
                        new_contexts[i].cpu()))
    return results
if __name__ == "__main__":
    model=ChessAttention()

    print("Total params:", sum(p.numel() for p in model.parameters()))
    
    board = chess.Board()
    root=MCTSNode(Node(board), None, 1.0, None, board.turn)
    mcts = MCTS(evaluator=batch_evaluator, num_workers=3)  # workers -> multiprocessing selection
    move, visits, _ = mcts.mp_search(root, num_sims=256)
    print("Best move:", move)
    print("Visit distribution:", {chess.Move.from_uci(uci).uci(): n for uci, n in visits.items()})
    mcts.close()
    

