#!/usr/bin/env python3
import argparse, math, random, pickle, os, time
import torch, chess, lmdb, tqdm, numpy as np
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional
import moves
from Model import ChessAttention

TEMPERATURE = 2.5
DEPTH_NOISE_END = 12
DEPTH_NOISE_SCALE = 0.8
MAX_PIECES = 33
CONTEXT_LENGTH = 2688
LMDB_PATH = './lmdb_data'
VIRTUAL_LOSS = 3.0
SAMPLE_MOVES = 8
os.makedirs(LMDB_PATH, exist_ok=True)

_lmdb_env = None
def get_env(write=False):
    global _lmdb_env
    if _lmdb_env is None or write:
        _lmdb_env = lmdb.open(LMDB_PATH, map_size=2**31, subdir=True, lock=True, readonly=not write)
    return _lmdb_env

def save_node_dict(node_dict):
    node_dict['ver'] = node_dict.get('ver', 0) + 1
    with get_env(write=True).begin(write=True) as txn:
        txn.put(f"node_{node_dict['id']}".encode(), pickle.dumps(node_dict))

def load_node_dict(node_id):
    with get_env(write=False).begin() as txn:
        data = txn.get(f"node_{node_id}".encode())
    return pickle.loads(data) if data else None

def _unpack_ctx(ctx_obj):
    if ctx_obj is None or isinstance(ctx_obj, torch.Tensor):
        return ctx_obj
    try:
        return pickle.loads(ctx_obj)
    except:
        return ctx_obj

def apply_dirichlet_noise(policy: Dict[str, float], alpha=0.25, epsilon=0.2):
    moves_list = list(policy.keys())
    if not moves_list:
        return policy
    noise = np.random.gamma(alpha, 1.0, len(moves_list))
    noise /= noise.sum() + 1e-10
    return {m: policy[m]*(1-epsilon) + float(n)*epsilon for m, n in zip(moves_list, noise)}

class Node:
    __slots__ = ('board', 'board_tensor', 'context')
    
    def __init__(self, board: chess.Board, context: Optional[torch.Tensor] = None):
        self.board = board
        self.board_tensor = self._to_tensor()
        if context is None:
            self.context = torch.zeros(CONTEXT_LENGTH, dtype=torch.float32)
        else:
            self.context = (context.clone().detach().float() if isinstance(context, torch.Tensor)
                           else torch.from_numpy(np.asarray(context, dtype=np.float32).flatten()[:CONTEXT_LENGTH]).float())

    def _to_tensor(self):
        tens = torch.zeros((MAX_PIECES, 4), dtype=torch.float32)
        piece_map = self.board.piece_map()
        for i, (sq, pc) in enumerate(piece_map.items()):
            if i >= MAX_PIECES - 1:
                break
            tens[i] = torch.tensor([pc.piece_type - 1, float(pc.color), (sq >> 3) - 3.5, (sq & 7) - 3.5])
        tens[MAX_PIECES-1, :2] = torch.tensor([10.0, float(self.board.ply())])
        return tens

_node_id_counter = mp.Value('i', 0)
def get_next_node_id():
    with _node_id_counter.get_lock():
        _node_id_counter.value += 1
        return _node_id_counter.value

class MCTSNode:
    __slots__ = ('id', 'state', 'parent_id', 'move', 'turn', 'prior', 'children', 
                 'N', 'W', 'W_anti', 'Q', 'antiQ', 'variance', 'virtual_loss', 
                 'is_expanded', 'opponent_softmax_weights')
    
    def __init__(self, state: Node, parent_id: Optional[int], prior: float, move_uci: Optional[str], turn: chess.Color):
        self.id = get_next_node_id()
        self.state = state
        self.parent_id = parent_id
        self.move = move_uci
        self.turn = turn
        self.prior = float(prior)
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.W_anti = 0.0
        self.Q = 0.0
        self.antiQ = 0.0
        self.variance = 1.0
        self.virtual_loss = 0
        self.is_expanded = False
        self.opponent_softmax_weights = None

    def to_dict(self):
        try:
            ctx_bytes = self.state.context.cpu().numpy().astype(np.float32).tobytes()
        except:
            ctx_bytes = None
        return {
            'id': self.id, 'state': self.state.board.fen(), 'move': self.move,
            'N': self.N, 'W': self.W, 'W_anti': self.W_anti, 'Q': self.Q, 'antiQ': self.antiQ,
            'variance': self.variance, 'is_expanded': self.is_expanded, 'children': self.children,
            'turn': int(self.turn), 'prior': self.prior, 'context': ctx_bytes,
            'virtual_loss': self.virtual_loss, 'Q_weights': self.opponent_softmax_weights
        }

def worker_process(root_id: int, out_queue: mp.Queue, shutdown_event: mp.Event, c_puct: float):
    wenv = get_env(write=False)
    while not shutdown_event.is_set():
        try:
            with wenv.begin() as txn:
                root_node = txn.get(f'node_{root_id}'.encode())
                if root_node is None:
                    time.sleep(0.005)
                    continue
                node = pickle.loads(root_node)
                path = [node['id']]
                root_turn = node['turn']
                depth = 0

                while node.get('is_expanded'):
                    child_ids = list(node['children'].values())
                    if not child_ids:
                        break
                    
                    children = [pickle.loads(txn.get(f'node_{cid}'.encode())) 
                               for cid in child_ids if txn.get(f'node_{cid}'.encode())]
                    if not children:
                        break

                    weights = torch.tensor(node.get('Q_weights', [1.0]*len(children)), dtype=torch.float32)
                    parent_N = max(1, sum(max(1, ch['N'] + ch.get('virtual_loss', 0)) for ch in children))
                    
                    if depth >= DEPTH_NOISE_END:
                        node = random.choice(children)
                        path.append(node['id'])
                        depth += 1
                        continue

                    scores = torch.zeros(len(children))
                    sqrt_parent = math.sqrt(parent_N)
                    depth_factor = max(0, depth - DEPTH_NOISE_END)
                    
                    for i, ch in enumerate(children):
                        n_virtual = ch['N'] + ch.get('virtual_loss', 0)
                        w_virtual = ch['W'] - VIRTUAL_LOSS * ch.get('virtual_loss', 0)
                        q_virtual = w_virtual / max(1, n_virtual)
                        
                        exploit = q_virtual if ch['turn'] == root_turn else ch['antiQ']
                        u = c_puct * ch['prior'] * sqrt_parent / (1 + n_virtual) * (1.0 + ch['variance'])
                        scores[i] = weights[i] * exploit + u
                        if depth_factor > 0:
                            scores[i] += random.gauss(0, DEPTH_NOISE_SCALE * depth_factor)
                    
                    if depth < SAMPLE_MOVES and len(children) > 1:
                        probs = torch.softmax(scores * 3.0, dim=0)
                        choice = torch.multinomial(probs, 1).item()
                        node = children[choice]
                    else:
                        node = children[scores.argmax().item()]
                    
                    nd_up = load_node_dict(node['id'])
                    if nd_up:
                        nd_up['virtual_loss'] = nd_up.get('virtual_loss', 0) + 1
                        save_node_dict(nd_up)
                    
                    path.append(node['id'])
                    depth += 1

                out_queue.put((node['id'], node['state'], node['context'], path))
        except:
            time.sleep(0.005)

def _expand_and_backup_shared(leaf_id, policy, v, av, var, ctx_bytes, path):
    leaf = load_node_dict(leaf_id)
    if leaf is None:
        return
    
    leaf['virtual_loss'] = max(0, leaf.get('virtual_loss', 0) - 1)
    
    if not leaf.get('is_expanded'):
        board = chess.Board(leaf['state'])
        ctx_tensor = _unpack_ctx(ctx_bytes)
        
        for uci, p in policy.items():
            try:
                move = chess.Move.from_uci(uci)
                new_board = board.copy(stack=False)
                new_board.push(move)
                child_node = Node(new_board, ctx_tensor)
                child = MCTSNode(child_node, leaf['id'], p, uci, not board.turn)
                leaf['children'][uci] = child.id
                save_node_dict(child.to_dict())
            except:
                continue
        
        leaf.update({'is_expanded': True, 'variance': var, 'W': v, 'W_anti': av, 'N': 1, 'Q': v, 'antiQ': av})
        save_node_dict(leaf)

    for nid in reversed(path):
        nd = load_node_dict(nid)
        if nd is None:
            continue
        nd['virtual_loss'] = max(0, nd.get('virtual_loss', 0) - 1)
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

def batch_evaluate(model, device, items: List[Tuple], temperature=2.5, use_amp=False):
    nodes, leaf_ids, paths = [], [], []
    for leaf_id, fen, ctx_bytes, path in items:
        ctx_tensor = None
        if ctx_bytes:
            ctx_tensor = torch.from_numpy(np.frombuffer(ctx_bytes, dtype=np.float32)).float()
        nodes.append(Node(chess.Board(fen), ctx_tensor))
        leaf_ids.append(leaf_id)
        paths.append(path)

    tens = torch.stack([n.board_tensor for n in nodes]).to(device)
    contexts = torch.stack([n.context for n in nodes]).to(device)
    
    with torch.no_grad():
        if use_amp:
            with torch.cuda.amp.autocast():
                values, antivalues, variances, policy_logits, new_contexts = model(tens, contexts)
        else:
            values, antivalues, variances, policy_logits, new_contexts = model(tens, contexts)

    results = []
    for i, node in enumerate(nodes):
        legal = list(node.board.legal_moves)
        policy = {}
        if legal:
            indices = [moves.pmove_to_idx.get(m.uci(), -1) for m in legal]
            logits = torch.full((len(legal),), float('-inf'), device=device)
            for j, idx in enumerate(indices):
                if 0 <= idx < policy_logits.size(1):
                    logits[j] = policy_logits[i, idx]
            
            if logits.isfinite().any():
                probs = torch.softmax(logits / max(1e-6, temperature), dim=0)
                policy = {m.uci(): float(p) for m, p in zip(legal, probs)}
            else:
                uniform = 1.0 / len(legal)
                policy = {m.uci(): uniform for m in legal}
        
        new_ctx_bytes = new_contexts[i].cpu().numpy().astype(np.float32).tobytes()
        results.append((leaf_ids[i], policy, float(values[i]), float(antivalues[i]), 
                       float(variances[i]), new_ctx_bytes, paths[i]))
    return results

def tpu_evaluator_loop(rank, manager_dict):
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    model = ChessAttention().to(device).eval()
    
    nodes_queue = manager_dict['nodes_queue']
    results_queue = manager_dict['results_queue']
    shutdown_event = manager_dict['shutdown_event']
    batch_size = manager_dict['batch_size']
    
    random.seed(time.time() + rank)
    
    while not shutdown_event.is_set():
        batch = []
        try:
            while len(batch) < batch_size:
                item = nodes_queue.get(timeout=0.3)
                batch.append(item)
        except:
            if shutdown_event.is_set() or not batch:
                if not batch:
                    continue
                else:
                    break
        
        results = batch_evaluate(model, device, batch, TEMPERATURE, use_amp=False)
        for res in results:
            results_queue.put(res)
        xm.mark_step()

def tpu_controller_loop(manager_dict):
    nodes_queue = manager_dict['nodes_queue']
    results_queue = manager_dict['results_queue']
    shutdown_event = manager_dict['shutdown_event']
    num_sims = manager_dict['num_sims']
    dirichlet_alpha = manager_dict['dirichlet_alpha']
    dirichlet_epsilon = manager_dict['dirichlet_epsilon']
    
    sims_done = 0
    bar = tqdm.tqdm(total=num_sims, ncols=80)
    
    try:
        while sims_done < num_sims:
            try:
                item = results_queue.get(timeout=1.0)
            except:
                if shutdown_event.is_set():
                    break
                continue
            
            leaf_id, policy, v, av, var, new_ctx_bytes, path = item
            
            if dirichlet_alpha and len(path) == 1:
                policy = apply_dirichlet_noise(policy, dirichlet_alpha, dirichlet_epsilon)
            
            _expand_and_backup_shared(leaf_id, policy, v, av, var, new_ctx_bytes, path)
            sims_done += 1
            bar.update(1)
    finally:
        bar.close()
        shutdown_event.set()

def tpu_main(rank, manager_dict):
    if rank == 0:
        import threading
        ctrl_thread = threading.Thread(target=lambda: tpu_controller_loop(manager_dict), daemon=True)
        ctrl_thread.start()
    
    try:
        tpu_evaluator_loop(rank, manager_dict)
    except KeyboardInterrupt:
        manager_dict['shutdown_event'].set()

class MCTS:
    def __init__(self, mode='gpu', num_workers=6, c_puct=2.0, dirichlet_alpha=0.25, 
                 dirichlet_epsilon=0.2, batch_size=24, num_sims=512):
        assert mode in ('gpu', 'tpu')
        self.mode = mode
        self.num_workers = max(1, num_workers)
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.batch_size = batch_size
        self.num_sims = num_sims
        self.eval_queue = None
        self.results_queue = None
        self.shutdown_event = None
        self.worker_procs = []

    def start_cpu_workers(self, root_id, out_queue, shutdown_event):
        for _ in range(self.num_workers):
            p = mp.Process(target=worker_process, args=(root_id, out_queue, shutdown_event, self.c_puct), daemon=True)
            p.start()
            self.worker_procs.append(p)

    def stop_cpu_workers(self):
        if self.shutdown_event:
            self.shutdown_event.set()
        for p in self.worker_procs:
            p.join(timeout=0.3)
        self.worker_procs = []

    def mp_search_gpu(self, root: MCTSNode):
        mgr = mp.Manager()
        self.eval_queue = mgr.Queue(maxsize=8192)
        self.results_queue = mgr.Queue(maxsize=8192)
        self.shutdown_event = mgr.Event()

        self.start_cpu_workers(root.id, self.eval_queue, self.shutdown_event)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ChessAttention().to(device).eval()

        try:
            save_node_dict(root.to_dict())
            sims_done = 0
            bar = tqdm.tqdm(total=self.num_sims, ncols=80)

            while sims_done < self.num_sims:
                items = []
                try:
                    timeout = 0.5
                    end_time = time.time() + timeout
                    while len(items) < self.batch_size and time.time() < end_time:
                        remaining = self.num_sims - sims_done - len(items)
                        if remaining <= 0:
                            break
                        try:
                            itm = self.eval_queue.get(timeout=0.1)
                            items.append(itm)
                        except:
                            break
                except:
                    continue
                
                if not items:
                    continue

                results = batch_evaluate(model, device, items, TEMPERATURE, use_amp=(device.type=='cuda'))
                
                for res in results:
                    leaf_id, policy, v, av, var, new_ctx_bytes, path = res
                    if self.dirichlet_alpha and len(path) == 1:
                        policy = apply_dirichlet_noise(policy, self.dirichlet_alpha, self.dirichlet_epsilon)
                    _expand_and_backup_shared(leaf_id, policy, v, av, var, new_ctx_bytes, path)
                    sims_done += 1
                    bar.update(1)

            bar.close()
        finally:
            self.shutdown_event.set()
            self.stop_cpu_workers()

        root_dict = load_node_dict(root.id)
        visits = {uci: load_node_dict(cid)['N'] for uci, cid in root_dict['children'].items()}
        best_uci = max(visits, key=visits.get)
        best_child = load_node_dict(root_dict['children'][best_uci])
        return best_uci, visits, best_child

    def mp_search_tpu(self, root: MCTSNode, nprocs=8):
        manager = mp.Manager()
        nodes_queue = manager.Queue(maxsize=8192)
        results_queue = manager.Queue(maxsize=8192)
        shutdown_event = manager.Event()

        self.start_cpu_workers(root.id, nodes_queue, shutdown_event)

        manager_dict = {
            'nodes_queue': nodes_queue,
            'results_queue': results_queue,
            'shutdown_event': shutdown_event,
            'num_sims': self.num_sims,
            'batch_size': self.batch_size,
            'c_puct': self.c_puct,
            'dirichlet_alpha': self.dirichlet_alpha,
            'dirichlet_epsilon': self.dirichlet_epsilon
        }

        save_node_dict(root.to_dict())

        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(tpu_main, args=(manager_dict,), nprocs=nprocs, start_method='fork')

        shutdown_event.set()
        self.stop_cpu_workers()

        root_dict = load_node_dict(root.id)
        visits = {uci: load_node_dict(cid)['N'] for uci, cid in root_dict['children'].items()}
        best_uci = max(visits, key=visits.get)
        best_child = load_node_dict(root_dict['children'][best_uci])
        return best_uci, visits, best_child

    def mp_search(self, root: MCTSNode):
        return self.mp_search_tpu(root) if self.mode == 'tpu' else self.mp_search_gpu(root)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['gpu','tpu'], default='gpu')
    p.add_argument('--num_sims', type=int, default=512)
    p.add_argument('--num_workers', type=int, default=6)
    p.add_argument('--batch_size', type=int, default=24)
    p.add_argument('--c_puct', type=float, default=2.0)
    return p.parse_args()

def main():
    args = parse_args()

    print("Clearing LMDB...")
    with get_env(write=True).begin(write=True) as txn:
        cursor = txn.cursor()
        for key, _ in cursor:
            txn.delete(key)

    root_board = chess.Board()
    root_node = MCTSNode(Node(root_board), None, 1.0, None, root_board.turn)
    print(f"Position:\n{root_board}\n")
    
    mcts = MCTS(mode=args.mode, num_workers=args.num_workers, 
                batch_size=args.batch_size, num_sims=args.num_sims, c_puct=args.c_puct)
    print(f"MCTS {args.mode.upper()}: {args.num_sims} sims, {args.num_workers} workers, batch={args.batch_size}, c_puct={args.c_puct}")
    
    move, visits, child = mcts.mp_search(root_node)
    print(f"\nBest move: {move}")
    print(f"Visits: {dict(sorted(visits.items(), key=lambda x: -x[1])[:10])}")
    print(f"Total sims: {sum(visits.values())}")

if __name__ == '__main__':
    main()
