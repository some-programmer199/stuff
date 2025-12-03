import math, random, pickle, time, os, threading
import torch, chess, lmdb, numpy as np
import multiprocessing as mp
from typing import List, Dict,Tuple, Optional, Any
import moves
# ====================
# CONFIG
# ====================
TEMPERATURE = 250
DEPTH_NOISE_END = 10
DEPTH_NOISE_SCALE = 50.0  # reduced from 100
MAX_PIECES = 33
CONTEXT_LENGTH = 256*4
LMDB_PATH = './lmdb_data'
os.makedirs(LMDB_PATH, exist_ok=True)

# ====================
# LMDB ENV
# ====================
env = lmdb.open(LMDB_PATH, map_size=2**30, max_dbs=0, subdir=True, lock=True)

# ====================
# HELPERS
# ====================
def _unpack_ctx(ctx_obj):
    if ctx_obj is None: return None
    if isinstance(ctx_obj, torch.Tensor): return ctx_obj
    try:
        return pickle.loads(ctx_obj)
    except Exception:
        return ctx_obj

def lmdb_put_node(node_dict):
    key = f"node_{node_dict['id']}".encode()
    with env.begin(write=True) as txn:
        txn.put(key, pickle.dumps(node_dict))

def lmdb_get_node(node_id):
    key = f"node_{node_id}".encode()
    with env.begin(write=False) as txn:
        data = txn.get(key)
    return pickle.loads(data) if data else None

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
            if i >= MAX_PIECES - 1: break
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
        lmdb_put_node(self.to_dict())

    def to_dict(self):
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
            'variance': self.variance,
            'is_expanded': self.is_expanded,
            'children': self.children,
            'turn': int(self.turn),
            'prior': self.prior,
            'context': ctx_pickled,
            'virtual_loss': self.virtual_loss,
            'Q_weights': self.opponent_softmax_weights
        }

# ====================
# WORKER PROCESS
# ====================
def worker_process(root_id: int, env_path: str, eval_queue: mp.Queue, shutdown_event: mp.Event, c_puct: float, dirichlet_alpha: float, dirichlet_epsilon: float):
    wenv = lmdb.open(env_path, readonly=True, lock=False)
    def get_node(txn, nid):
        data = txn.get(f'node_{nid}'.encode())
        return pickle.loads(data) if data else None

    local_cache = {}
    while not shutdown_event.is_set():
        try:
            with wenv.begin() as txn:
                root_node = get_node(txn, root_id)
                if root_node is None: time.sleep(0.01); continue
                node = root_node
                path = [node['id']]
                root_turn = node['turn']
                depth = 0

                # traverse tree
                while node.get('is_expanded') and node.get('children'):
                    children = []
                    for cid in node['children'].values():
                        if cid in local_cache:
                            children.append(local_cache[cid])
                        else:
                            ch = get_node(txn, cid)
                            if ch: 
                                local_cache[cid] = ch
                                children.append(ch)
                    if not children: break

                    weights = node.get('Q_weights')
                    weights = torch.tensor(weights, dtype=torch.float32) if weights else torch.ones(len(children))
                    parent_N = max(1, sum(max(1, ch.get('N',0)) for ch in children))
                    best_score = -float('inf')
                    best_idx = 0
                    depth_factor = max(0, depth - DEPTH_NOISE_END)
                    for i, ch in enumerate(children):
                        exploit = ch.get('Q',0.0) if ch.get('turn',1)!=root_turn else ch.get('antiQ',0.0)
                        u = c_puct * ch.get('prior',0.0) * math.sqrt(parent_N)/(1+ch.get('N',0)) * (1.0 + ch.get('variance',1.0))
                        score = float(weights[i]) * exploit + u
                        if depth_factor>0:
                            score += random.gauss(0, DEPTH_NOISE_SCALE*depth_factor)
                        if score>best_score:
                            best_score = score
                            best_idx = i
                    node = children[best_idx]
                    path.append(node['id'])
                    depth +=1

                leaf_id = node['id']
                eval_queue.put((leaf_id, node['state'], node['context'], path))
        except Exception as e:
            time.sleep(0.01)
            continue

# ====================
# MCTS CONTROLLER
# ====================
class MCTS:
    def __init__(self, evaluator=None, c_puct=2.5, dirichlet_alpha=0.3, dirichlet_epsilon=0.25, batch_size=16, num_workers=4):
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.batch_size = batch_size
        self.num_workers = max(1,num_workers)
        self.lock = mp.Lock()
        self._eval_queue: Optional[mp.Queue] = None
        self._shutdown_event: Optional[mp.Event] = None
        self._workers: List[mp.Process] = []

    # Placeholder for evaluator batch function
    def _batch_eval_nodes(self, nodes: List[Node]) -> List[Any]:
        if self.evaluator is None:
            return [({},0.0,0.0,1.0,pickle.dumps(n.context)) for n in nodes]
        return self.evaluator(nodes)

    # Orchestrate MP search
    def mp_search(self, root: MCTSNode, num_sims:int=512):
        # evaluate root + apply dirichlet noise
        policy, v, av, var, ctx = self._batch_eval_nodes([root.state])[0]
        if self.dirichlet_alpha:
            moves_list = list(policy.keys())
            if moves_list:
                noise = np.random.gamma(self.dirichlet_alpha,1.0,len(moves_list))
                noise /= noise.sum()
                policy = {m: policy[m]*(1-self.dirichlet_epsilon)+float(n)*self.dirichlet_epsilon for m,n in zip(moves_list,noise)}

        # expand root children
        board = root.state.board
        for uci,p in policy.items():
            move = chess.Move.from_uci(uci)
            new_board = board.copy(stack=False)
            new_board.push(move)
            child_node = Node(new_board, _unpack_ctx(ctx))
            child = MCTSNode(child_node, root.id, p, uci, not board.turn)
            root.children[uci] = child.id
        root.is_expanded = True
        lmdb_put_node(root.to_dict())

        # spawn workers
        self._eval_queue = mp.Queue(maxsize=1024)
        self._shutdown_event = mp.Event()
        for _ in range(self.num_workers):
            p = mp.Process(target=worker_process,
                           args=(root.id, LMDB_PATH, self._eval_queue, self._shutdown_event, self.c_puct,self.dirichlet_alpha,self.dirichlet_epsilon))
            p.start()
            self._workers.append(p)

        sims_done = 0
        while sims_done<num_sims:
            batch=[]
            try:
                while len(batch)<self.batch_size:
                    batch.append(self._eval_queue.get(timeout=0.2))
            except Exception:
                pass
            if not batch: continue

            # deduplicate
            unique={}
            for leaf_id, fen, ctx_pickled, path in batch:
                unique[leaf_id]=(fen,ctx_pickled,path)
            eval_nodes=[]
            leaf_ids=[]
            paths=[]
            for leaf_id,(fen,ctx_pickled,path) in list(unique.items())[:self.batch_size]:
                eval_nodes.append(Node(chess.Board(fen), _unpack_ctx(ctx_pickled)))
                leaf_ids.append(leaf_id)
                paths.append(path)
            results=self._batch_eval_nodes(eval_nodes)

            # backup
            for i,(leaf_id,(fen,ctx_pickled,path)) in enumerate(list(unique.items())[:self.batch_size]):
                policy,v,av,var,ctx_p=results[i]
                self._expand_and_backup(leaf_id, policy,v,av,var,ctx_p,path)
                sims_done+=1

        # shutdown workers
        self._shutdown_event.set()
        for p in self._workers: p.join(timeout=1.0)
        self._workers=[]

        # pick best move
        root_dict = lmdb_get_node(root.id)
        visits={uci: lmdb_get_node(cid)['N'] for uci,cid in root_dict['children'].items()}
        best_uci = max(visits,key=visits.get)
        best_child = lmdb_get_node(root_dict['children'][best_uci])
        return best_uci, visits, best_child

    def _expand_and_backup(self, leaf_id, policy, v, av, var, ctx_pickled, path):
        with self.lock:
            leaf = lmdb_get_node(leaf_id)
            if leaf is None: return
            if not leaf.get('is_expanded'):
                board = chess.Board(leaf['state'])
                for uci,p in policy.items():
                    move = chess.Move.from_uci(uci)
                    new_board = board.copy(stack=False)
                    new_board.push(move)
                    child_node = Node(new_board,_unpack_ctx(ctx_pickled))
                    child = MCTSNode(child_node, None, p, uci, not board.turn)
                    leaf['children'][uci]=child.id
                leaf['is_expanded']=True
                leaf['variance']=var
                leaf['W']=v; leaf['W_anti']=av; leaf['N']=1
                leaf['Q']=v; leaf['antiQ']=av
                lmdb_put_node(leaf)

            # backup along path
            for nid in reversed(path):
                nd=lmdb_get_node(nid)
                if nd is None: continue
                nd['N']+=1; nd['W']+=v; nd['W_anti']+=av
                nd['Q']=nd['W']/nd['N']; nd['antiQ']=nd['W_anti']/nd['N']; nd['variance']=var
                lmdb_put_node(nd)
    def close(self):
        if self._shutdown_event:
            self._shutdown_event.set()
        for p in self._workers:
            p.join(timeout=1.0)
        self._workers=[]
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
    

