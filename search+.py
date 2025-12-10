import math, random, pickle, os, time
import torch, chess, lmdb, tqdm, numpy as np
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional, Any
import moves
from Model import ChessAttention

# ====================
# CONFIG
# ====================
TEMPERATURE = 3.0
DEPTH_NOISE_END = 10
DEPTH_NOISE_SCALE = 1.0
MAX_PIECES = 33
CONTEXT_LENGTH = 256 * 4
LMDB_PATH = './lmdb_data'
os.makedirs(LMDB_PATH, exist_ok=True)
#delete existing LMDB for fresh start
if os.path.exists(LMDB_PATH):
    for file in os.listdir(LMDB_PATH):
        file_path = os.path.join(LMDB_PATH, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Error deleting file {file_path}: {e}')
# ====================
# LMDB ENV FACTORY
# ====================
def get_env(write=False):
    return lmdb.open(
        LMDB_PATH,
        map_size=2**30,
        subdir=True,
        lock=True,
        max_dbs=0,
        readonly=not write
    )

def save_node_dict(node_dict):
    node_dict['ver'] = node_dict.get('ver', 0) + 1
    key = f"node_{node_dict['id']}".encode()
    with get_env(write=True).begin(write=True) as txn:
        txn.put(key, pickle.dumps(node_dict))

def load_node_dict(node_id):
    key = f"node_{node_id}".encode()
    with get_env(write=False).begin() as txn:
        data = txn.get(key)
    return pickle.loads(data) if data else None

# ====================
# HELPERS
# ====================
def _unpack_ctx(ctx_obj):
    if ctx_obj is None:
        return None
    if isinstance(ctx_obj, torch.Tensor):
        return ctx_obj
    try:
        return pickle.loads(ctx_obj)
    except Exception:
        return ctx_obj

def apply_dirichlet_noise(policy: Dict[str, float], alpha: float = 0.3, epsilon: float = 0.25):
    moves_list = list(policy.keys())
    if not moves_list:
        return policy
    noise = np.random.gamma(alpha, 1.0, len(moves_list))
    noise /= noise.sum()
    return {m: policy[m]*(1-epsilon) + float(n)*epsilon for m, n in zip(moves_list, noise)}

# ====================
# NODE
# ====================
class Node:
    def __init__(self, board: chess.Board, context: Optional[torch.Tensor] = None):
        self.board = board
        self.board_tensor = self._to_tensor()
        if context is None:
            self.context = torch.zeros(CONTEXT_LENGTH, dtype=torch.float32)
        else:
            self.context = context.clone().detach().float() if isinstance(context, torch.Tensor) else torch.from_numpy(np.asarray(context, dtype=np.float32).flatten()[:CONTEXT_LENGTH]).float()

    def _to_tensor(self):
        tens = torch.zeros((MAX_PIECES, 4), dtype=torch.float32)
        for i, (sq, pc) in enumerate(self.board.piece_map().items()):
            if i >= MAX_PIECES - 1:
                break
            tens[i, 0] = pc.piece_type - 1
            tens[i, 1] = float(pc.color)
            tens[i, 2] = (sq // 8) - 3.5
            tens[i, 3] = (sq % 8) - 3.5
        tens[MAX_PIECES-1, 0] = 10.0
        tens[MAX_PIECES-1, 1] = float(self.board.ply())
        return tens

# ====================
# ATOMIC NODE ID
# ====================
_node_id_counter = mp.Value('i', 0)
def get_next_node_id():
    with _node_id_counter.get_lock():
        _node_id_counter.value += 1
        return _node_id_counter.value

# ====================
# MCTS NODE
# ====================
class MCTSNode:
    def __init__(self, state: Node, parent_id: Optional[int], prior: float, move_uci: Optional[str], turn: chess.Color):
        self.id = get_next_node_id()
        self.state = state
        self.parent_id = parent_id
        self.move = move_uci
        self.turn = turn
        self.prior = float(prior)
        self.children: Dict[str, int] = {}
        self.N = 0
        self.W = 0.0
        self.W_anti = 0.0
        self.Q = 0.0
        self.antiQ = 0.0
        self.variance = 1.0
        self.virtual_loss = 0
        self.is_expanded = False
        self.opponent_softmax_weights: Optional[List[float]] = None

    def to_dict(self):
        try:
            ctx_bytes = self.state.context.cpu().numpy().astype(np.float32).tobytes()
        except Exception:
            ctx_bytes = None
        return {
            'id': self.id,
            'state': self.state.board.fen(),
            'move': self.move,
            'N': self.N,
            'W': self.W,
            'W_anti': self.W_anti,
            'Q': self.Q,
            'antiQ': self.antiQ,
            'variance': self.variance,
            'is_expanded': self.is_expanded,
            'children': self.children,
            'turn': int(self.turn),
            'prior': self.prior,
            'context': ctx_bytes,
            'virtual_loss': self.virtual_loss,
            'Q_weights': self.opponent_softmax_weights
        }

# ====================
# WORKER PROCESS
# ====================
def worker_process(root_id: int, eval_queue: mp.Queue, shutdown_event: mp.Event, c_puct: float):
    wenv = get_env(write=False)
    while not shutdown_event.is_set():
        try:
            with wenv.begin() as txn:
                root_node = txn.get(f'node_{root_id}'.encode())
                if root_node is None:
                    time.sleep(0.01)
                    continue
                root_node = pickle.loads(root_node)
                node = root_node
                path = [node['id']]
                root_turn = node['turn']
                depth = 0

                while node.get('is_expanded'):
                    children = []
                    for cid in node['children'].values():
                        ch_data = txn.get(f'node_{cid}'.encode())
                        if ch_data:
                            ch = pickle.loads(ch_data)
                            children.append(ch)
                    if not children:
                        break

                    weights = node.get('Q_weights')
                    weights = torch.tensor(weights, dtype=torch.float32) if weights else torch.ones(len(children))
                    parent_N = max(1, sum(max(1, ch.get('N',0)) for ch in children))
                    best_score = -float('inf')
                    best_idx = 0
                    depth_factor = max(0, depth - DEPTH_NOISE_END)
                    if depth_factor == DEPTH_NOISE_END:
                        node = random.choice(children)
                        path.append(node['id'])
                    for i, ch in enumerate(children):
                        exploit = ch['Q'] if ch['turn']==root_turn else ch['antiQ']
                        u = c_puct * ch.get('prior',0.0) * math.sqrt(parent_N)/(1+ch.get('N',0)) * (1.0 + ch.get('variance',1.0))
                        score = float(weights[i]) * exploit + u
                        if depth_factor > 0:
                            score += random.gauss(0, DEPTH_NOISE_SCALE*depth_factor)
                        if score > best_score:
                            best_score = score
                            best_idx = i
                    node = children[best_idx]
                    path.append(node['id'])
                    depth += 1

                leaf_id = node['id']
                eval_queue.put((leaf_id, node['state'], node['context'], path))
        except Exception:
            time.sleep(0.01)
            continue
    else:
        print("Shutdown event received.")
    print("Worker shutting down.")
# ====================
# MCTS CONTROLLER
# ====================
class MCTS:
    def __init__(self, evaluator=None, c_puct=2.5, dirichlet_alpha=0.3, dirichlet_epsilon=0.25, batch_size=16, num_workers=4, shutdown_event: Optional[mp.Event]=None):
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.batch_size = batch_size
        self.num_workers = max(1, num_workers)
        self.lock = mp.Lock()
        self._eval_queue: Optional[mp.Queue] = None
        self._shutdown_event = shutdown_event
        self._workers: List[mp.Process] = []

    def _batch_eval_nodes(self, nodes: List[Node]) -> List[Any]:
        if self.evaluator is None:
            return [({},0.0,0.0,1.0,pickle.dumps(n.context)) for n in nodes]
        return self.evaluator(nodes)

    def mp_search(self, root: MCTSNode, num_sims:int=512):
      try:
        save_node_dict(root.to_dict())
        policy, v, av, var, ctx = self._batch_eval_nodes([root.state])[0]
        if self.dirichlet_alpha:
            policy = apply_dirichlet_noise(policy, self.dirichlet_alpha, self.dirichlet_epsilon)

        # Expand root children
        board = root.state.board
        for uci, p in policy.items():
            move = chess.Move.from_uci(uci)
            new_board = board.copy(stack=False)
            new_board.push(move)
            child_node = Node(new_board, _unpack_ctx(ctx))
            child = MCTSNode(child_node, root.id, p, uci, not board.turn)
            root.children[uci] = child.id
            save_node_dict(child.to_dict())
        root.is_expanded = True
        save_node_dict(root.to_dict())

        # Spawn workers
        self._eval_queue = mp.Queue(maxsize=1024)
        self._shutdown_event = mp.Event()
        for _ in range(self.num_workers):
            p = mp.Process(target=worker_process,
                           args=(root.id, self._eval_queue, self._shutdown_event, self.c_puct),daemon=True)
            p.start()
            self._workers.append(p)

        bar = tqdm.tqdm(total=num_sims)
        sims_done = 0
        while sims_done < num_sims or num_sims-sims_done > self.batch_size:
            if sims_done <= 16:
                self.batch_size=64
            batch = []
            try:
                while len(batch) < self.batch_size or sims_done + len(batch) < num_sims:
                    batch.append(self._eval_queue.get(timeout=0.5))
            except Exception:
                continue
            if not batch:
                continue

            unique = {}
            for leaf_id, fen, ctx_bytes, path in batch:
                unique[leaf_id] = (fen, ctx_bytes, path)

            eval_nodes, leaf_ids, paths = [], [], []
            for leaf_id, (fen, ctx_bytes, path) in list(unique.items())[:self.batch_size]:
                ctx_tensor = torch.frombuffer(np.frombuffer(ctx_bytes, dtype=np.float32), dtype=torch.float32) if ctx_bytes else None
                eval_nodes.append(Node(chess.Board(fen), ctx_tensor))
                leaf_ids.append(leaf_id)
                paths.append(path)

            results = self._batch_eval_nodes(eval_nodes)

            # Backup
            for i, leaf_id in enumerate(leaf_ids):
                policy, v, av, var, ctx_p = results[i]
                self._expand_and_backup(leaf_id, policy, v, av, var, ctx_p, paths[i])
                sims_done += 1
                bar.update(1)
        bar.close()
      finally:

        # Pick best move
        root_dict = load_node_dict(root.id)
        visits = {uci: load_node_dict(cid)['N'] for uci, cid in root_dict['children'].items()}
        best_uci = max(visits, key=visits.get)
        best_child = load_node_dict(root_dict['children'][best_uci])
        return best_uci, visits, best_child

    def _expand_and_backup(self, leaf_id, policy, v, av, var, ctx_pickled, path):
        with self.lock:
            leaf = load_node_dict(leaf_id)
            if leaf is None:
                return
            if not leaf.get('is_expanded'):
                board = chess.Board(leaf['state'])
                for uci, p in policy.items():
                    move = chess.Move.from_uci(uci)
                    new_board = board.copy(stack=False)
                    new_board.push(move)
                    child_node = Node(new_board, _unpack_ctx(ctx_pickled))
                    child = MCTSNode(child_node, leaf['id'], p, uci, not board.turn)
                    leaf['children'][uci] = child.id
                    save_node_dict(child.to_dict())
                leaf['is_expanded'] = True
                leaf['variance'] = var
                leaf['W'] = v
                leaf['W_anti'] = av
                leaf['N'] = 1
                leaf['Q'] = v
                leaf['antiQ'] = av
                save_node_dict(leaf)

            # Backup along path
            for nid in reversed(path):
                nd = load_node_dict(nid)
                if nd is None:
                    continue
                nd['N'] += 1
                if nd['turn'] == leaf['turn']:
                    nd['W'] += v
                    nd['W_anti'] += av
                else:
                    nd['W'] += av
                    nd['W_anti'] += v
                nd['Q'] = nd['W'] / nd['N']
                nd['antiQ'] = nd['W_anti'] / nd['N']
                nd['variance'] = var
                save_node_dict(nd)

    def close(self):
        if self._shutdown_event:
            print("Shutting down MCTS workers...")
            self._shutdown_event.set()
            print(self._shutdown_event.is_set())
            print("Waiting for workers to terminate...")
        for p in self._workers:
            p.join(timeout=0.1)
            print("Worker terminated.")
        self._workers = []
        print("All workers shut down.")

# ====================
# BATCH EVALUATOR
# ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ChessAttention()
model.to(device)
model.eval()
print("compiling model...")
model.compile()
print("model compiled.")
def batch_evaluator(nodes: List[Node]) -> List[Tuple[dict, float, float, float, torch.Tensor]]:
    tens = torch.stack([n.board_tensor for n in nodes], dim=0).to('cpu')
    contexts = torch.stack([n.context for n in nodes], dim=0).to('cpu')
    values, antivalues, variances, policy_logits, new_contexts = model(tens, contexts)
    results = []
    for i, node in enumerate(nodes):
        board = node.board
        legal = list(board.legal_moves)
        policy = {}
        if legal:
            logits = torch.full((len(legal),), float('-inf'))
            for j, m in enumerate(legal):
                idx = moves.pmove_to_idx.get(m.uci(), -1)
                if 0 <= idx < policy_logits.size(1):
                    logits[j] = float(policy_logits[i, idx].detach().cpu())
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

# ====================
# MAIN
# ====================
if __name__ == "__main__":
    board = chess.Board()
    root = MCTSNode(Node(board), None, 1.0, None, board.turn)
    shutdown_event = mp.Event()
    mcts = MCTS(evaluator=batch_evaluator, num_workers=3,shutdown_event=shutdown_event)
    move, visits, _ = mcts.mp_search(root, num_sims=256)
    print("Best move:", move)
    print("Visit distribution:", {chess.Move.from_uci(uci).uci(): n for uci, n in visits.items()})
    print(sum(visits.values()), "simulations performed.")
    mcts.close()
    

