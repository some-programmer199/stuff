import torch
import chess
import math
import numpy as np
from model import ChessAttention, HISTORY_LEN, HISTORY_PAD_IDX
import moves
import tqdm
import multiprocessing as mp
from multiprocessing import Queue, Process, Value, Array
import ctypes
import time
from queue import Empty, Full

# ---------------- Config ----------------
CONTEXT_LENGTH = 2688
MAX_PIECES = 33
C_PUCT = 2.0
VIRTUAL_LOSS = 3.0
TEMPERATURE = 2.5
NUM_WORKERS = 4  # Start with fewer workers
BATCH_SIZE = 16  # Smaller batch size to start
DIRICHLET_ALPHA = 0.3
NOISE_EPS = 0.25
MAX_CHILDREN_EST = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Shared Arrays using multiprocessing.Array ----------------
def create_shared_arrays(max_nodes):
    """Create shared arrays using multiprocessing.Array (more reliable than shared_memory)"""
    arrays = {
        'N': Array(ctypes.c_int32, max_nodes, lock=False),
        'W': Array(ctypes.c_float, max_nodes, lock=False),
        'Wanti': Array(ctypes.c_float, max_nodes, lock=False),
        'Q': Array(ctypes.c_float, max_nodes, lock=False),
        'antiQ': Array(ctypes.c_float, max_nodes, lock=False),
        'VAR': Array(ctypes.c_float, max_nodes, lock=False),
        'PRIOR': Array(ctypes.c_float, max_nodes, lock=False),
        'TURN': Array(ctypes.c_bool, max_nodes, lock=False),
        'VLOSS': Array(ctypes.c_int16, max_nodes, lock=False),
        'EXP': Array(ctypes.c_bool, max_nodes, lock=False),
        'PARENT': Array(ctypes.c_int32, max_nodes, lock=False),
        'CH_PTR': Array(ctypes.c_int32, max_nodes, lock=False),
        'CH_LEN': Array(ctypes.c_int16, max_nodes, lock=False),
        'CH_BUF': Array(ctypes.c_int32, max_nodes * MAX_CHILDREN_EST, lock=False),
        'QW_BUF': Array(ctypes.c_float, max_nodes * MAX_CHILDREN_EST, lock=False),
        'NODE_LOCKS': Array(ctypes.c_uint8, max_nodes, lock=False),
        'MOVE_BUF': Array(ctypes.c_int32, max_nodes, lock=False),
    }
    
    # Initialize
    for i in range(max_nodes):
        arrays['VAR'][i] = 1.0
        arrays['PARENT'][i] = -1
        arrays['MOVE_BUF'][i] = -1
    
    for i in range(max_nodes * MAX_CHILDREN_EST):
        arrays['CH_BUF'][i] = -1
    
    return arrays

def create_shared_tensors(max_nodes):
    """Create large tensor arrays separately"""
    return mp.RawArray(ctypes.c_uint16, max_nodes * CONTEXT_LENGTH)

# ---------------- Worker Globals ----------------
N = W = Wanti = Q = antiQ = VAR = PRIOR = TURN = VLOSS = EXP = None
PARENT = CH_PTR = CH_LEN = CH_BUF = QW_BUF = NODE_LOCKS = None
MOVE_BUF = None
CTX_RAW = None
CTX_VIEW = None
ROOT_FEN = None
MAX_NODES = 0

def init_worker_arrays(shared_arrays, ctx_raw, root_fen, max_nodes):
    """Initialize arrays in worker process"""
    global N, W, Wanti, Q, antiQ, VAR, PRIOR, TURN, VLOSS, EXP
    global PARENT, CH_PTR, CH_LEN, CH_BUF, QW_BUF, NODE_LOCKS, MOVE_BUF
    global CTX_RAW, CTX_VIEW, ROOT_FEN, MAX_NODES
    
    N = shared_arrays['N']
    W = shared_arrays['W']
    Wanti = shared_arrays['Wanti']
    Q = shared_arrays['Q']
    antiQ = shared_arrays['antiQ']
    VAR = shared_arrays['VAR']
    PRIOR = shared_arrays['PRIOR']
    TURN = shared_arrays['TURN']
    VLOSS = shared_arrays['VLOSS']
    EXP = shared_arrays['EXP']
    PARENT = shared_arrays['PARENT']
    CH_PTR = shared_arrays['CH_PTR']
    CH_LEN = shared_arrays['CH_LEN']
    CH_BUF = shared_arrays['CH_BUF']
    QW_BUF = shared_arrays['QW_BUF']
    NODE_LOCKS = shared_arrays['NODE_LOCKS']
    MOVE_BUF = shared_arrays['MOVE_BUF']
    CTX_RAW = ctx_raw
    ROOT_FEN = root_fen
    MAX_NODES = max_nodes
    CTX_VIEW = np.frombuffer(CTX_RAW, dtype=np.float16).reshape(MAX_NODES, CONTEXT_LENGTH)

# ---------------- Helper Functions ----------------
def get_ctx(nid):
    """Get context for a node"""
    return CTX_VIEW[nid]

def set_ctx(nid, ctx):
    """Set context for a node"""
    CTX_VIEW[nid] = np.asarray(ctx, dtype=np.float16)

def reconstruct_board(nid):
    board = chess.Board(ROOT_FEN)
    seq = []
    cur = nid
    while cur > 0:
        mv_idx = MOVE_BUF[cur]
        if mv_idx < 0:
            break
        seq.append(mv_idx)
        cur = PARENT[cur]
    for mv_idx in reversed(seq):
        board.push(chess.Move.from_uci(moves.pmoves[mv_idx]))
    return board

def board_to_tensor(board):
    tens = np.zeros((MAX_PIECES, 4), dtype=np.float32)
    piece_items = board.piece_map().items()
    for i, (sq, pc) in enumerate(piece_items):
        if i >= MAX_PIECES - 1:
            break
        tens[i, 0] = pc.piece_type - 1
        tens[i, 1] = float(pc.color)
        tens[i, 2] = (sq >> 3) - 3.5
        tens[i, 3] = (sq & 7) - 3.5
    tens[MAX_PIECES - 1, 0] = 10.0
    tens[MAX_PIECES - 1, 1] = float(board.ply())
    return tens

def board_to_history(board):
    history = np.full(HISTORY_LEN, HISTORY_PAD_IDX, dtype=np.int64)
    move_stack = list(board.move_stack)[-HISTORY_LEN:]
    start = HISTORY_LEN - len(move_stack)
    for i, mv in enumerate(move_stack):
        history[start + i] = moves.pmove_to_idx.get(mv.uci(), HISTORY_PAD_IDX)
    return history

def alloc_node(node_counter, node_lock):
    with node_lock:
        nid = node_counter.value
        if nid >= MAX_NODES:
            return -1
        node_counter.value += 1
        return nid

def alloc_children(k, child_counter, child_lock):
    with child_lock:
        ptr = child_counter.value
        max_children = MAX_NODES * MAX_CHILDREN_EST
        if ptr + k > max_children:
            return -1
        child_counter.value += k
        return ptr

def acquire_node(nid):
    """Simple spinlock"""
    max_attempts = 1000
    for attempt in range(max_attempts):
        if NODE_LOCKS[nid] == 0:
            NODE_LOCKS[nid] = 1
            return True
        time.sleep(0.0001)
    return False

def release_node(nid):
    NODE_LOCKS[nid] = 0

def summarize_profile(label, timings, count):
    if count <= 0:
        return
    total = sum(timings.values())
    parts = []
    for key, value in timings.items():
        pct = (value / total * 100.0) if total > 0 else 0.0
        parts.append(f"{key}={value:.3f}s ({pct:.1f}%)")
    detail = ", ".join(parts)
    avg_ms = (total / count * 1000.0) if count > 0 else 0.0
    print(f"{label} profile: steps={count}, total={total:.3f}s, avg={avg_ms:.2f}ms, {detail}")

def add_dirichlet_noise(policy, rng, eps=NOISE_EPS, alpha=DIRICHLET_ALPHA):
    if not policy:
        return policy
    moves_list = list(policy.keys())
    noise = rng.dirichlet([alpha] * len(moves_list))
    mixed = {}
    for mv, n in zip(moves_list, noise):
        mixed[mv] = (1 - eps) * policy[mv] + eps * float(n)
    return mixed

# ---------------- Tree Operations ----------------
def select_path(root):
    path = [root]
    nid = root
    
    while EXP[nid] and CH_LEN[nid] > 0:
        ptr = CH_PTR[nid]
        k = CH_LEN[nid]
        
        # Safely get children
        kids = np.frombuffer(CH_BUF, dtype=np.int32, count=k, offset=ptr * np.dtype(np.int32).itemsize).copy()

        sqrt_parent = math.sqrt(N[nid] + 1)
        n_vals = np.fromiter((N[k] + VLOSS[k] for k in kids), dtype=np.float32, count=k)
        w_vals = np.fromiter((W[k] - VIRTUAL_LOSS * VLOSS[k] for k in kids), dtype=np.float32, count=k)
        wanti_vals = np.fromiter((Wanti[k] for k in kids), dtype=np.float32, count=k)
        turn_vals = np.fromiter((TURN[k] for k in kids), dtype=np.bool_, count=k)
        prior_vals = np.fromiter((PRIOR[k] for k in kids), dtype=np.float32, count=k)
        var_vals = np.fromiter((VAR[k] for k in kids), dtype=np.float32, count=k)
        qw_vals = np.frombuffer(QW_BUF, dtype=np.float32, count=k, offset=ptr * np.dtype(np.float32).itemsize)

        denom = np.maximum(n_vals, 1.0)
        q_vals = w_vals / denom
        aq_vals = wanti_vals / denom
        value = np.where(turn_vals == TURN[nid], q_vals, aq_vals)
        exploit = qw_vals * value
        explore = C_PUCT * prior_vals * sqrt_parent / (1.0 + n_vals) * var_vals
        best_idx = int(np.argmax(exploit + explore))

        # Select best child
        idx = best_idx
        nid = kids[idx]
        VLOSS[nid] += 1
        path.append(nid)
    
    return nid, path

def expand(nid, policy, var, node_counter, node_lock, child_counter, child_lock):
    if not acquire_node(nid):
        return
    
    try:
        if EXP[nid]:
            return
        
        board = reconstruct_board(nid)
        legal = list(board.legal_moves)
        if not legal:
            EXP[nid] = True
            return
        
        ptr = alloc_children(len(legal), child_counter, child_lock)
        if ptr < 0:
            return
        CH_PTR[nid] = ptr
        CH_LEN[nid] = len(legal)
        
        for i, mv in enumerate(legal):
            cid = alloc_node(node_counter, node_lock)
            if cid < 0:
                CH_LEN[nid] = i
                break
            CH_BUF[ptr + i] = cid
            
            PARENT[cid] = nid
            TURN[cid] = not board.turn
            PRIOR[cid] = policy.get(mv.uci(), 0.0)
            MOVE_BUF[cid] = moves.pmove_to_idx.get(mv.uci(), -1)
        
        update_weights(nid)
        EXP[nid] = True
        VAR[nid] = var
    finally:
        release_node(nid)

def update_weights(nid, temp=1.0):
    ptr = CH_PTR[nid]
    k = CH_LEN[nid]
    if k <= 0:
        return
    
    opp_values = np.empty(k, dtype=np.float32)

    for i in range(k):
        kid = CH_BUF[ptr + i]
        n = N[kid] + VLOSS[kid]
        w = W[kid] - VIRTUAL_LOSS * VLOSS[kid]
        q = w / max(n, 1)
        aq = Wanti[kid] / max(n, 1)
        
        opp_val = aq if TURN[kid] == TURN[nid] else q
        opp_values[i] = opp_val
    
    # Softmax
    scaled = opp_values / (temp + 1e-12)
    exp_vals = np.exp(scaled - np.max(scaled))
    weights = exp_vals / np.sum(exp_vals)
    
    for i in range(k):
        QW_BUF[ptr + i] = weights[i]

def backup(path, leaf, v, av, var):
    for nid in path:
        N[nid] += 1
        if TURN[nid] == TURN[leaf]:
            W[nid] += v
            Wanti[nid] += av
        else:
            W[nid] += av
            Wanti[nid] += v
        
        Q[nid] = W[nid] / N[nid]
        antiQ[nid] = Wanti[nid] / N[nid]
        VAR[nid] = var
        VLOSS[nid] = max(0, VLOSS[nid] - 1)
    
    # Update weights for parents
    parent_ids = set()
    for nid in path:
        pid = PARENT[nid]
        if pid >= 0 and CH_LEN[pid] > 0:
            parent_ids.add(pid)
    for pid in parent_ids:
        update_weights(pid)

# ---------------- Search Worker ----------------
def search_worker(worker_id, shared_arrays, ctx_raw, root_fen, max_nodes,
                  node_counter, node_lock, child_counter, child_lock,
                  eval_queue, result_queue, stop_flag, root_node, sims_target, eps=NOISE_EPS, alpha=DIRICHLET_ALPHA, use_anti=True):
    try:
        init_worker_arrays(shared_arrays, ctx_raw, root_fen, max_nodes)
        print(f"Worker {worker_id} started")
        rng = np.random.default_rng(worker_id + int(time.time()))
        timings = {
            "select": 0.0,
            "enqueue": 0.0,
            "wait_result": 0.0,
            "expand": 0.0,
            "backup": 0.0,
        }
        iterations = 0
        req_id = 0
        in_flight = {}
        max_in_flight = 2
        
        while not stop_flag.value:
            try:
                loop_progress = False
                while True:
                    t0 = time.perf_counter()
                    try:
                        result = result_queue.get_nowait()
                    except Empty:
                        timings["wait_result"] += time.perf_counter() - t0
                        break
                    timings["wait_result"] += time.perf_counter() - t0
                    loop_progress = True
                    if result is None:
                        stop_flag.value = 1
                        break
                    done_req_id, nid, policy, v, av, var, new_ctx = result
                    path = in_flight.pop(done_req_id, None)
                    if path is None:
                        continue
                    policy = add_dirichlet_noise(policy, rng, eps=eps, alpha=alpha)
                    set_ctx(nid, new_ctx)
                    t0 = time.perf_counter()
                    expand(nid, policy, var, node_counter, node_lock, child_counter, child_lock)
                    timings["expand"] += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    backup(path, nid, v, av, var)
                    timings["backup"] += time.perf_counter() - t0
                    iterations += 1

                while len(in_flight) < max_in_flight and not stop_flag.value:
                    if N[root_node] + len(in_flight) >= sims_target:
                        break
                    t0 = time.perf_counter()
                    leaf, path = select_path(root_node)
                    timings["select"] += time.perf_counter() - t0
                    if EXP[leaf]:
                        for pid in path:
                            VLOSS[pid] = max(0, VLOSS[pid] - 1)
                        continue
                    t0 = time.perf_counter()
                    try:
                        eval_queue.put_nowait((worker_id, req_id, leaf))
                        in_flight[req_id] = path
                        req_id += 1
                        loop_progress = True
                    except Full:
                        for pid in path:
                            VLOSS[pid] = max(0, VLOSS[pid] - 1)
                        break
                    timings["enqueue"] += time.perf_counter() - t0

                if not loop_progress:
                    time.sleep(0.001)
                
            except Exception as e:
                print(f"Worker {worker_id} iteration error: {e}")
                continue
        
        print(f"Worker {worker_id} stopped")
        summarize_profile(f"Worker {worker_id}", timings, iterations)
    except Exception as e:
        print(f"Worker {worker_id} fatal error: {e}")
        import traceback
        traceback.print_exc()

# ---------------- Evaluation Worker ----------------
def evaluation_worker(shared_arrays, ctx_raw, root_fen, max_nodes,
                      eval_queue, result_queues, stop_flag, batch_size=BATCH_SIZE):
    try:
        init_worker_arrays(shared_arrays, ctx_raw, root_fen, max_nodes)
        
        print("Loading model...")
        model = ChessAttention().to(device).eval()
        print(f"Model loaded on {device}")
        
        pending_evals = []
        timings = {
            "collect": 0.0,
            "tensor": 0.0,
            "model": 0.0,
            "policy": 0.0,
            "send": 0.0,
        }
        batches = 0
        
        while not stop_flag.value or not eval_queue.empty():
            try:
                # Collect batch
                timeout = 0.01 if pending_evals else 0.1
                try:
                    t0 = time.perf_counter()
                    item = eval_queue.get(timeout=timeout)
                    timings["collect"] += time.perf_counter() - t0
                    if item is None:
                        break
                    pending_evals.append(item)
                except:
                    pass
                
                # Process batch
                if len(pending_evals) >= batch_size or (pending_evals and eval_queue.empty()):
                    batch = pending_evals[:batch_size]
                    pending_evals = pending_evals[batch_size:]
                    
                    worker_ids = [item[0] for item in batch]
                    req_ids = [item[1] for item in batch]
                    leaf_nodes = [item[2] for item in batch]
                    
                    # Get tensors
                    t0 = time.perf_counter()
                    boards_for_nodes = [reconstruct_board(nid) for nid in leaf_nodes]
                    boards_list = [board_to_tensor(board) for board in boards_for_nodes]
                    history_list = [board_to_history(board) for board in boards_for_nodes]
                    
                    boards_np = np.stack(boards_list, axis=0)
                    history_np = np.stack(history_list, axis=0)
                    boards_torch = torch.from_numpy(boards_np).to(device)
                    history_torch = torch.from_numpy(history_np).to(device)
                    history_mask = history_torch.eq(HISTORY_PAD_IDX)
                    timings["tensor"] += time.perf_counter() - t0
                    
                    # Evaluate
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        v, av, var, logits, new_ctxs = model(
                            boards_torch,
                            history_torch,
                            history_mask
                        )
                    timings["model"] += time.perf_counter() - t0
                    
                    # Process results
                    t0 = time.perf_counter()
                    results = []
                    for i, nid in enumerate(leaf_nodes):
                        board = boards_for_nodes[i]
                        legal = list(board.legal_moves)
                        policy = {}
                        
                        if legal:
                            idxs = torch.tensor(
                                [moves.pmove_to_idx.get(m.uci(), -1) for m in legal],
                                dtype=torch.long,
                                device=device,
                            )
                            valid = (idxs >= 0) & (idxs < logits.size(1))
                            l = torch.full((len(legal),), -1e9, device=device)
                            if torch.any(valid):
                                l[valid] = logits[i, idxs[valid]]
                            probs = torch.softmax(l / TEMPERATURE, dim=0)
                            policy = {m.uci(): float(p) for m, p in zip(legal, probs)}
                        
                        results.append((
                            nid,
                            policy,
                            float(v[i]),
                            float(av[i]),
                            float(var[i]),
                            new_ctxs[i].cpu().numpy()
                        ))
                    timings["policy"] += time.perf_counter() - t0
                    
                    # Send results
                    t0 = time.perf_counter()
                    for worker_id, req_id, result in zip(worker_ids, req_ids, results):
                        result_queues[worker_id].put((req_id, *result))
                    timings["send"] += time.perf_counter() - t0
                    batches += 1
                        
            except Exception as e:
                print(f"Eval worker batch error: {e}")
                continue
        
        # Send stop signals
        for q in result_queues:
            q.put(None)
        
        print("Evaluation worker stopped")
        summarize_profile("Eval worker", timings, batches)
    except Exception as e:
        print(f"Eval worker fatal error: {e}")
        import traceback
        traceback.print_exc()

# ---------------- Main MCTS ----------------
def run_mcts_parallel(board, sims=800, num_workers=NUM_WORKERS, eps=NOISE_EPS, alpha=DIRICHLET_ALPHA, use_anti=True):
    print("Initializing shared memory...")
    max_nodes = max(4096, sims * MAX_CHILDREN_EST + num_workers * 32 + 1)
    shared_arrays = create_shared_arrays(max_nodes)
    ctx_raw = create_shared_tensors(max_nodes)
    
    node_counter = Value('i', 1)
    node_lock = mp.Lock()
    child_counter = Value('i', 0)
    child_lock = mp.Lock()
    
    # Initialize root
    root = 0
    root_fen = board.fen()
    
    # Access raw arrays through numpy
    ctx_arr = np.frombuffer(ctx_raw, dtype=np.float16).reshape(max_nodes, CONTEXT_LENGTH)
    
    ctx_arr[root] = 0
    shared_arrays['TURN'][root] = board.turn
    shared_arrays['PRIOR'][root] = 1.0
    
    eval_queue = Queue(maxsize=num_workers * 4)
    result_queues = [Queue(maxsize=4) for _ in range(num_workers)]
    stop_flag = Value('i', 0)
    
    print("Starting evaluation worker...")
    eval_proc = Process(
        target=evaluation_worker,
        args=(shared_arrays, ctx_raw, root_fen, max_nodes, eval_queue, result_queues, stop_flag)
    )
    eval_proc.start()
    
    time.sleep(2)  # Give eval worker time to load model
    
    print(f"Starting {num_workers} search workers...")
    workers = []
    for i in range(num_workers):
        p = Process(
            target=search_worker,
            args=(i, shared_arrays, ctx_raw, root_fen, max_nodes,
                  node_counter, node_lock, child_counter, child_lock,
                  eval_queue, result_queues[i], stop_flag, root, sims, eps, alpha, use_anti)
        )
        p.start()
        workers.append(p)
    
    print(f"Running {sims} simulations...")
    pbar = tqdm.tqdm(total=sims, desc="Simulations", position=0)
    node_bar = tqdm.tqdm(total=max_nodes, desc="Nodes created", position=1, leave=False)
    last_count = 0
    last_nodes = node_counter.value
    
    try:
        while True:
            current_count = shared_arrays['N'][root]
            if current_count >= sims:
                break
            
            delta = current_count - last_count
            if delta > 0:
                pbar.update(delta)
                last_count = current_count

            current_nodes = node_counter.value
            delta_nodes = current_nodes - last_nodes
            if delta_nodes > 0:
                node_bar.update(delta_nodes)
                last_nodes = current_nodes
            
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        pbar.close()
        node_bar.close()
    
    print("Stopping workers...")
    stop_flag.value = 1
    
    for w in workers:
        w.join(timeout=5)
        if w.is_alive():
            w.terminate()
    
    eval_proc.join(timeout=5)
    if eval_proc.is_alive():
        eval_proc.terminate()
    
    # Get best move
    ptr = shared_arrays['CH_PTR'][root]
    k = shared_arrays['CH_LEN'][root]
    kids = [shared_arrays['CH_BUF'][ptr + i] for i in range(k)]
    visits = [shared_arrays['N'][kid] for kid in kids]
    best_idx = visits.index(max(visits))
    best = kids[best_idx]
    
    print(f"\nTotal visits: {shared_arrays['N'][root]}")
    print(f"Nodes created: {node_counter.value}")
    
    best_move_idx = shared_arrays['MOVE_BUF'][best]
    return best, visits, kids, best_move_idx

# ---------------- Example Usage ----------------
if __name__ == "__main__":
    mp.set_start_method('spawn')
    
    board = chess.Board()
    best, visits, kids, best_move_idx = run_mcts_parallel(board, sims=200, num_workers=3)
    print("Best Move:", moves.pmoves[best_move_idx])
    print("Visits:", visits[:10])
