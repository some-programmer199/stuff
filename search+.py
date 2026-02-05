#!/usr/bin/env python3
"""
mcts_tpu_gpu.py
Run MCTS with CPU selection workers and either:
 - GPU mode: single-process evaluator on CUDA/CPU
 - TPU mode: full 8-core v5e evaluation using torch_xla/xmp.spawn

Requirements:
 - chess, lmdb, torch, tqdm, numpy
 - Model.py with ChessAttention and moves.pmove_to_idx mapping
 - For TPU mode: torch_xla available on the environment
"""

import argparse
import math, random, pickle, os, time
import torch, chess, lmdb, tqdm, numpy as np
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional, Any
import moves
from Model import ChessAttention

# -----------------------
# CONFIG (tweak as desired)
# -----------------------
TEMPERATURE = 3.0
DEPTH_NOISE_END = 10
DEPTH_NOISE_SCALE = 1.0
MAX_PIECES = 33
CONTEXT_LENGTH = 256 * 4
LMDB_PATH = './lmdb_data'
os.makedirs(LMDB_PATH, exist_ok=True)

# -----------------------
# LMDB helpers
# -----------------------
def get_env(write=False):
    # subdir True makes LMDB create a directory
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

# -----------------------
# helpers
# -----------------------
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

# -----------------------
# Node & MCTSNode
# -----------------------
class Node:
    def __init__(self, board: chess.Board, context: Optional[torch.Tensor] = None):
        self.board = board
        self.board_tensor = self._to_tensor()
        if context is None:
            self.context = torch.zeros(CONTEXT_LENGTH, dtype=torch.float32)
        else:
            self.context = (context.clone().detach().float()
                            if isinstance(context, torch.Tensor)
                            else torch.from_numpy(np.asarray(context, dtype=np.float32).flatten()[:CONTEXT_LENGTH]).float())

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

_node_id_counter = mp.Value('i', 0)
def get_next_node_id():
    with _node_id_counter.get_lock():
        _node_id_counter.value += 1
        return _node_id_counter.value

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

# -----------------------
# CPU selection worker (unchanged logic, but pushes to a manager queue)
# -----------------------
def worker_process(root_id: int, out_queue: mp.Queue, shutdown_event: mp.Event, c_puct: float):
    """
    CPU process: reads LMDB tree, traverses down using PUCT until a leaf, then emits
    (leaf_id, fen, context_bytes, path_list_of_ids) into out_queue for evaluation.
    """
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

                # descend until an unexpanded leaf is found
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
                # push the work to evaluation queue
                out_queue.put((leaf_id, node['state'], node['context'], path))
        except Exception:
            # keep worker alive; sleep a little to avoid tight loop on transient errors
            time.sleep(0.01)
            continue
    # cleanup
    # print("Worker shutting down.")  # noisy in many workers

# -----------------------
# Local GPU/CPU batch evaluator (used in GPU mode)
# -----------------------
def batch_evaluator_local(model, device, items: List[Tuple[int,str,Optional[bytes],List[int]]], temperature=3.0):
    """
    items: list of tuples (leaf_id, fen, ctx_bytes, path)
    returns list of tuples: (leaf_id, policy, v, av, var, new_ctx_bytes, path)
    """
    nodes = []
    leaf_ids = []
    paths = []
    for leaf_id, fen, ctx_bytes, path in items:
        ctx_tensor = None
        if ctx_bytes:
            arr = np.frombuffer(ctx_bytes, dtype=np.float32)
            ctx_tensor = torch.from_numpy(arr).float()
        nodes.append(Node(chess.Board(fen), ctx_tensor))
        leaf_ids.append(leaf_id)
        paths.append(path)

    tens = torch.stack([n.board_tensor for n in nodes], dim=0).to(device)
    contexts = torch.stack([n.context for n in nodes], dim=0).to(device)
    with torch.no_grad():
        values, antivalues, variances, policy_logits, new_contexts = model(tens, contexts)

    results = []
    for i in range(len(nodes)):
        board = nodes[i].board
        legal = list(board.legal_moves)
        policy = {}
        if legal:
            logits = torch.full((len(legal),), float('-inf'), device=device)
            for j, m in enumerate(legal):
                idx = moves.pmove_to_idx.get(m.uci(), -1)
                if 0 <= idx < policy_logits.size(1):
                    logits[j] = float(policy_logits[i, idx].detach().cpu())
            if logits.isfinite().any():
                probs = torch.softmax(logits / max(1e-6, temperature), dim=0)
                policy = {m.uci(): p for m, p in zip(legal, probs.tolist())}
            else:
                policy = {m.uci(): 1.0 / len(legal) for m in legal}
        # convert new_contexts[i] to bytes for storage/transfer
        new_ctx_bytes = new_contexts[i].cpu().numpy().astype(np.float32).tobytes()
        results.append((leaf_ids[i], policy,
                        float(values[i].item()),
                        float(antivalues[i].item()),
                        float(variances[i].item()),
                        new_ctx_bytes,
                        paths[i]))
    return results

# -----------------------
# TPU evaluator main (this function will be run under xmp.spawn in TPU mode)
# rank==0 will act as the controller (collecting results and doing backups),
# ranks 0..7 will also evaluate batches from nodes_queue.
# -----------------------
def tpu_main(rank, manager_dict):
    """
    manager_dict contains proxies created by mp.Manager() and configuration values
    This function runs in each spawned TPU process (ranks 0..nprocs-1).
    Rank 0 does controller duties; all ranks act as evaluators consuming nodes_queue.
    """
    # import XLA only inside spawned processes (avoid XLA init before forking other CPU workers)
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    # build model on each TPU rank
    model = ChessAttention().to(device).eval()

    # unpack manager proxies
    nodes_queue = manager_dict['nodes_queue']
    results_queue = manager_dict['results_queue']
    shutdown_event = manager_dict['shutdown_event']
    # search control params (only used by rank 0 controller)
    num_sims = manager_dict['num_sims']
    batch_size = manager_dict['batch_size']
    c_puct = manager_dict['c_puct']
    dirichlet_alpha = manager_dict['dirichlet_alpha']
    dirichlet_epsilon = manager_dict['dirichlet_epsilon']

    # slight random seed differentiation
    random.seed(time.time() + rank)

    # Evaluator loop (all ranks run this; rank 0 additionally runs controller logic)
    # We'll keep loops simple: evaluators form batches from nodes_queue and run model,
    # then send results back to results_queue for the controller to apply backups.
    def evaluator_loop():
        while not shutdown_event.is_set():
            batch = []
            try:
                # collect at most batch_size items
                while len(batch) < batch_size:
                    item = nodes_queue.get(timeout=0.5)
                    batch.append(item)  # (leaf_id, fen, ctx_bytes, path)
            except Exception:
                if shutdown_event.is_set():
                    break
                if not batch:
                    continue
            if not batch:
                continue

            # build Node objects on this rank and evaluate
            nodes = []
            leaf_ids = []
            paths = []
            for leaf_id, fen, ctx_bytes, path in batch:
                ctx_tensor = None
                if ctx_bytes:
                    arr = np.frombuffer(ctx_bytes, dtype=np.float32)
                    ctx_tensor = torch.from_numpy(arr).float()
                nodes.append(Node(chess.Board(fen), ctx_tensor))
                leaf_ids.append(leaf_id)
                paths.append(path)

            tens = torch.stack([n.board_tensor for n in nodes], dim=0).to(device)
            contexts = torch.stack([n.context for n in nodes], dim=0).to(device)
            with torch.no_grad():
                values, antivalues, variances, policy_logits, new_contexts = model(tens, contexts)

            # package results and push back
            for i in range(len(nodes)):
                board = nodes[i].board
                legal = list(board.legal_moves)
                policy = {}
                if legal:
                    logits = torch.full((len(legal),), float('-inf'), device=device)
                    for j, m in enumerate(legal):
                        idx = moves.pmove_to_idx.get(m.uci(), -1)
                        if 0 <= idx < policy_logits.size(1):
                            logits[j] = float(policy_logits[i, idx].detach().cpu())
                    if logits.isfinite().any():
                        probs = torch.softmax(logits / max(1e-6, TEMPERATURE), dim=0)
                        policy = {m.uci(): p for m, p in zip(legal, probs.tolist())}
                    else:
                        policy = {m.uci(): 1.0 / len(legal) for m in legal}

                new_ctx_bytes = new_contexts[i].cpu().numpy().astype(np.float32).tobytes()
                results_queue.put((leaf_ids[i], policy,
                                   float(values[i].item()),
                                   float(antivalues[i].item()),
                                   float(variances[i].item()),
                                   new_ctx_bytes,
                                   paths[i]))
            # flush XLA graph for predictable execution
            xm.mark_step()

    # Controller loop (only executed by rank 0)
    def controller_loop():
        # controller will collect results and apply expansion/backups
        sims_done = 0
        total_sims = num_sims
        bar = tqdm.tqdm(total=total_sims)
        try:
            while sims_done < total_sims:
                try:
                    # collect one result at a time and apply
                    item = results_queue.get(timeout=1.0)
                except Exception:
                    if shutdown_event.is_set():
                        break
                    continue
                # item == (leaf_id, policy, v, av, var, new_ctx_bytes, path)
                leaf_id, policy, v, av, var, new_ctx_bytes, path = item
                # optionally add dirichlet noise only when expanding root (we detect root by path containing root id 1? better: check root id saved)
                # here we will apply noise if requested and path length == 1 (i.e. root)
                if dirichlet_alpha and len(path) == 1:
                    policy = apply_dirichlet_noise(policy, dirichlet_alpha, dirichlet_epsilon)

                # expand and backup using LMDB
                _expand_and_backup_shared(leaf_id, policy, v, av, var, new_ctx_bytes, path)
                sims_done += 1
                bar.update(1)
        finally:
            bar.close()
            # finished
            shutdown_event.set()

    # run evaluator loop in this rank, and if rank==0 also run controller loop concurrently
    # the simplest approach here is: if rank==0, spawn a small thread that runs controller_loop
    if rank == 0:
        import threading
        ctrl_thread = threading.Thread(target=controller_loop, daemon=True)
        ctrl_thread.start()
    # all ranks run evaluator loop (including rank 0)
    try:
        evaluator_loop()
    except KeyboardInterrupt:
        shutdown_event.set()
    # if rank 0, wait for controller to finish
    if rank == 0:
        # controller may already be finishing
        time.sleep(0.1)
    # exit

# -----------------------
# Shared expand & backup that uses LMDB; used by controller (inside TPU ranks or GPU main)
# This is a module-level helper so it can be called from inside the spawned processes.
# -----------------------
def _expand_and_backup_shared(leaf_id, policy, v, av, var, ctx_pickled, path):
    """
    leaf_id: id of node (existing leaf)
    policy: dict move->prob
    v,av,var: floats
    ctx_pickled: bytes of float32 context vector
    path: list of node ids from root down to leaf (as saved in LMDB)
    """
    # locking with LMDB is per-txn; we still use a process-global file lock via mp.Lock if desired.
    # For simplicity, we just do LMDB read/write inside txn blocks (LMDB provides locking).
    # Load leaf
    leaf = load_node_dict(leaf_id)
    if leaf is None:
        return
    if not leaf.get('is_expanded'):
        board = chess.Board(leaf['state'])
        for uci, p in policy.items():
            try:
                move = chess.Move.from_uci(uci)
            except Exception:
                continue
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

    # Now backup along path (path is list of node ids from root -> ... -> leaf)
    # Note: path must contain IDs saved in LMDB (the selection worker produces these)
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

# -----------------------
# MCTS controller class (provides GPU mode orchestration; TPU mode uses xmp.spawn + controller inside spawned rank 0)
# -----------------------
class MCTS:
    def __init__(self,
                 mode: str = 'gpu',
                 num_workers: int = 4,
                 c_puct: float = 2.5,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25,
                 batch_size: int = 16,
                 num_sims: int = 512):
        assert mode in ('gpu', 'tpu')
        self.mode = mode
        self.num_workers = max(1, num_workers)
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.batch_size = batch_size
        self.num_sims = num_sims

        # local process queues (used by GPU mode)
        self.eval_queue: Optional[mp.Queue] = None
        self.results_queue: Optional[mp.Queue] = None
        self.shutdown_event = None
        self.worker_procs: List[mp.Process] = []

    def start_cpu_workers(self, root_id, out_queue, shutdown_event):
        # spawn selection workers (CPU-only)
        for _ in range(self.num_workers):
            p = mp.Process(target=worker_process, args=(root_id, out_queue, shutdown_event, self.c_puct), daemon=True)
            p.start()
            self.worker_procs.append(p)

    def stop_cpu_workers(self):
        if self.shutdown_event:
            self.shutdown_event.set()
        for p in self.worker_procs:
            p.join(timeout=0.5)
        self.worker_procs = []

    def mp_search_gpu(self, root: MCTSNode):
        """
        Runs selection workers (CPU), local evaluator (GPU/CPU), and expansion/backups in main process.
        """
        # create queues and shutdown event
        mgr = mp.Manager()
        self.eval_queue = mgr.Queue(maxsize=4096)
        self.results_queue = mgr.Queue(maxsize=4096)
        self.shutdown_event = mgr.Event()

        # start CPU selection workers; they push items into eval_queue
        self.start_cpu_workers(root.id, self.eval_queue, self.shutdown_event)

        # create model locally (GPU or CPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ChessAttention().to(device).eval()

        try:
            save_node_dict(root.to_dict())
            sims_done = 0
            bar = tqdm.tqdm(total=self.num_sims)

            # main loop: collect batches of work from eval_queue, run batch_evaluator_local, then expand & backup
            while sims_done < self.num_sims:
                items = []
                try:
                    while len(items) < self.batch_size and sims_done + len(items) < self.num_sims:
                        itm = self.eval_queue.get(timeout=1.0)  # (leaf_id, fen, ctx_bytes, path)
                        items.append(itm)
                except Exception:
                    continue
                if not items:
                    continue

                # evaluate locally
                results = batch_evaluator_local(model, device, items, temperature=TEMPERATURE)
                # results are tuples (leaf_id, policy, v, av, var, new_ctx_bytes, path)
                for res in results:
                    leaf_id, policy, v, av, var, new_ctx_bytes, path = res
                    # apply dirichlet if the leaf is the root (path includes root id)
                    if self.dirichlet_alpha and len(path) == 1:
                        policy = apply_dirichlet_noise(policy, self.dirichlet_alpha, self.dirichlet_epsilon)
                    _expand_and_backup_shared(leaf_id, policy, v, av, var, new_ctx_bytes, path)
                    sims_done += 1
                    bar.update(1)

            bar.close()
        finally:
            # shutdown workers
            self.shutdown_event.set()
            self.stop_cpu_workers()

        # pick best move
        root_dict = load_node_dict(root.id)
        visits = {uci: load_node_dict(cid)['N'] for uci, cid in root_dict['children'].items()}
        best_uci = max(visits, key=visits.get)
        best_child = load_node_dict(root_dict['children'][best_uci])
        return best_uci, visits, best_child

    def mp_search_tpu(self, root: MCTSNode, nprocs: int = 8):
        """
        Use xmp.spawn to launch nprocs TPU processes. The controller logic runs in rank 0 inside tpu_main,
        while all ranks (including rank 0) evaluate jobs from nodes_queue.
        CPU selection workers are started here (they push to the manager nodes_queue).
        """
        # manager for shared queues/events that can be passed to spawn children
        manager = mp.Manager()
        nodes_queue = manager.Queue(maxsize=4096)
        results_queue = manager.Queue(maxsize=4096)
        shutdown_event = manager.Event()

        # start CPU selection workers that will push to nodes_queue
        self.start_cpu_workers(root.id, nodes_queue, shutdown_event)

        # pack a small dict of manager proxies and config to pass to spawned processes
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

        # save root to LMDB before spawn so selection workers see it
        save_node_dict(root.to_dict())

        # xmp.spawn blocks until all spawned processes exit. We rely on the controller inside rank 0 to set shutdown_event when finished.
        import torch_xla.distributed.xla_multiprocessing as xmp
        # spawn the TPU ranks; they will import torch_xla inside tpu_main function
        xmp.spawn(tpu_main, args=(manager_dict,), nprocs=nprocs, start_method='fork')

        # after spawn returns, children have finished, shutdown_event is set
        # stop cpu workers
        shutdown_event.set()
        self.stop_cpu_workers()

        # choose best move
        root_dict = load_node_dict(root.id)
        visits = {uci: load_node_dict(cid)['N'] for uci, cid in root_dict['children'].items()}
        best_uci = max(visits, key=visits.get)
        best_child = load_node_dict(root_dict['children'][best_uci])
        return best_uci, visits, best_child

    def mp_search(self, root: MCTSNode):
        if self.mode == 'gpu':
            return self.mp_search_gpu(root)
        else:
            return self.mp_search_tpu(root, nprocs=8)

# -----------------------
# CLI & main
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['gpu','tpu'], default='gpu', help='Run mode: gpu or tpu (v5e-8).')
    p.add_argument('--num_sims', type=int, default=256)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=16)
    return p.parse_args()

def main():
    args = parse_args()

    # clean LMDB dir for fresh run (optional)
    print("Clearing LMDB data...")
    with get_env(write=True).begin(write=True) as txn:
        cursor = txn.cursor()
        for key, _ in cursor:
            txn.delete(key)
    print("LMDB data cleared.")
    # initial root node
    root_board = chess.Board()
    root_node = MCTSNode(Node(root_board), None, 1.0, None, root_board.turn)
    print("Initial position:\n", root_board)
    mcts = MCTS(mode=args.mode, num_workers=args.num_workers, batch_size=args.batch_size, num_sims=args.num_sims)
    print(f"Starting MCTS in {args.mode.upper()} mode: {args.num_sims} sims, {args.num_workers} CPU workers, batch_size={args.batch_size}")
    move, visits, child = mcts.mp_search(root_node)
    print("Best move:", move)
    print("Visit distribution:", {chess.Move.from_uci(uci).uci(): n for uci, n in visits.items()})
    print(sum(visits.values()), "simulations performed.")

if __name__ == '__main__':
    main()
