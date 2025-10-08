import chess
import torch
import torch.nn as nn
import math
import datetime
from multiprocessing import Process, Manager
import threading
import queue
import moves
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Inference server (separate process owns GPU) ----------
def inference_worker(request_queue, device_str):
    import torch
    from Bot import azt
    torch.set_num_threads(1)
    dev = torch.device(device_str)
    model = azt().to(dev)
    model.eval()
    while True:
        item = request_queue.get()
        if item is None:
            break
        batch_numpy, response_queue = item
        with torch.no_grad():
            batch_tensor = torch.from_numpy(batch_numpy).float().to(dev)
            Ps, Vs, CPs = model.forward(batch_tensor)
            Ps_np = Ps.detach().cpu().numpy()
            Vs_np = Vs.detach().cpu().numpy().reshape(-1)
            CPs_np = CPs.detach().cpu().numpy()
        response_queue.put((Ps_np, Vs_np, CPs_np))
    return

class InferenceServer:
    def __init__(self, device_str=None):
        self.manager = Manager()
        self.request_queue = self.manager.Queue()
        self.device_str = device_str or ("cuda" if torch.cuda.is_available() else "cpu")
        self.process = Process(target=inference_worker, args=(self.request_queue, self.device_str))
        self.process.daemon = True
        self.started = False
    def start(self):
        if not self.started:
            self.process.start()
            self.started = True
    def stop(self):
        if self.started:
            self.request_queue.put(None)
            self.process.join(timeout=5)
            self.started = False
    def infer(self, batch_tensor: torch.Tensor, timeout=None):
        if not self.started:
            self.start()
        batch_numpy = batch_tensor.detach().cpu().numpy()
        response_q = self.manager.Queue()
        self.request_queue.put((batch_numpy, response_q))
        result = response_q.get(timeout=timeout)
        return result

_inference_server = None
def get_inference_server():
    global _inference_server
    if _inference_server is None:
        _inference_server = InferenceServer(device_str=("cuda" if torch.cuda.is_available() else "cpu"))
        _inference_server.start()
    return _inference_server

# Evaluation speed stats (thread-safe)
_eval_stats = {
    "batches": 0,
    "positions": 0,
    "total_time": 0.0,
    "batch_times": []
}
_stats_lock = threading.Lock()

def reset_eval_stats():
    with _stats_lock:
        _eval_stats["batches"] = 0
        _eval_stats["positions"] = 0
        _eval_stats["total_time"] = 0.0
        _eval_stats["batch_times"] = []

def get_eval_stats():
    with _stats_lock:
        batches = _eval_stats["batches"]
        positions = _eval_stats["positions"]
        total_time = _eval_stats["total_time"]
        avg_batch_time = (total_time / batches) if batches > 0 else 0.0
        throughput = (positions / total_time) if total_time > 0 else 0.0
        return {
            "batches": batches,
            "positions": positions,
            "total_time": total_time,
            "avg_batch_time": avg_batch_time,
            "throughput_pos_per_sec": throughput
        }

# ---------- Chess Bot and MCTS ----------
class Bot:
    def __init__(self, simulations=100, num_cores=4):
        self.simulations = simulations
        self.num_cores = num_cores
        get_inference_server()
    def choose_move(self, root):
        bestmove, distribution = MCTSsearch(root, self.simulations, num_workers=self.num_cores)
        return bestmove.move, distribution

def movetoNN(move: chess.Move):
    movestr = str(move)
    # decide which move bucket: normal move (from-square) or promotion
    if len(movestr) == 4:
        fromsq = movestr[:2]
        try:
            start_idx, end_idx = moves.sq_idx[fromsq]
        except Exception:
            return None
    else:
        try:
            start_idx, end_idx = moves.sq_idx["promotion"]
        except Exception:
            return None
    # compare string forms (moves.pmoves contains move strings)
    for i, pmove in enumerate(moves.pmoves[start_idx:end_idx]):
        if pmove == movestr:
            return i + start_idx
    return None

class node:
    def __init__(self, board: chess.Board, tensor: torch.Tensor, parent, move):
        self.board = board
        self.tensor = tensor
        self.visits = 1
        self.move = move
        self.children = []
        self.is_expanded = False
        self.P, self.value, _ = [0] * 1858, 0, 0
        self.Q = self.value
    def __repr__(self):
        return str(self.move)
    def extend(self):
        for move in self.board.legal_moves:
            boardx = self.board.copy()
            boardx.push(move)
            self.children.append(node(boardx, self.tenspush(move), self, move))
        self.is_expanded = True
    def choosechild(self, cpuct=1.5):
        best_child = None
        best_value = -math.inf
        for child in self.children:
            Q = child.Q
            idx = movetoNN(child.move)
            prob = 0.0
            if idx is not None and 0 <= idx < len(self.P):
                prob = self.P[idx]
            a = child.visits + 1
            U = cpuct * prob * math.sqrt(self.visits) / a
            v = Q + U
            if v >= best_value:
                best_child = child
                best_value = v
        return best_child
    def tenspush(self, move: chess.Move):
        tensor = self.tensor.clone()
        fromsq = move.from_square
        tosq = move.to_square
        fromr = 7 - chess.square_rank(fromsq)
        fromc = chess.square_file(fromsq)
        tor = 7 - chess.square_rank(tosq)
        toc = chess.square_file(tosq)
        piece = self.board.piece_at(fromsq)
        if piece is None or tensor is None:
            return tensor
        if move.promotion:
            pawn_plane = piece_to_plane[chess.PAWN] + (0 if piece.color == chess.WHITE else 6)
            tensor[0, pawn_plane, fromr, fromc] = 0.0
            promo_plane = piece_to_plane[move.promotion] + (0 if piece.color == chess.WHITE else 6)
            tensor[0, promo_plane, tor, toc] = 1.0
            return tensor
        if self.board.is_castling(move):
            king_plane = piece_to_plane[chess.KING] + (0 if piece.color == chess.WHITE else 6)
            tensor[0, king_plane, fromr, fromc] = 0.0
            tensor[0, king_plane, tor, toc] = 1.0
            if toc > fromc:
                rook_fromc = 7
                rook_toc = 5
            else:
                rook_fromc = 0
                rook_toc = 3
            rook_row = fromr
            rook_plane = piece_to_plane[chess.ROOK] + (0 if piece.color == chess.WHITE else 6)
            tensor[0, rook_plane, rook_row, rook_fromc] = 0.0
            tensor[0, rook_plane, rook_row, rook_toc] = 1.0
            return tensor
        if self.board.is_en_passant(move):
            pawn_plane = piece_to_plane[chess.PAWN] + (0 if piece.color == chess.WHITE else 6)
            tensor[0, pawn_plane, fromr, fromc] = 0.0
            tensor[0, pawn_plane, tor, toc] = 1.0
            cap_row = tor + (1 if piece.color == chess.WHITE else -1)
            tensor[0, pawn_plane, cap_row, toc] = 0.0
            return tensor
        plane = piece_to_plane[piece.piece_type] + (0 if piece.color == chess.WHITE else 6)
        tensor[0, plane, fromr, fromc] = 0.0
        tensor[0, plane, tor, toc] = 1.0
        return tensor
    def update_Q(self, value):
        self.visits += 1
        self.Q += (value - self.Q) / self.visits

piece_to_plane = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

def crunch_board(boardx: chess.Board):
    tensorend = torch.zeros((0, 8, 8), dtype=torch.float32)
    for x in range(8):
        tensor = torch.zeros(13, 8, 8)
        for square in chess.SQUARES:
            piece = boardx.piece_at(square)
            if piece:
                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)
                base = 0 if piece.color == chess.WHITE else 6
                plane = base + piece_to_plane[piece.piece_type]
                tensor[plane, row, col] = 1.0
        tensor[12, :, :] = float(boardx.ply())
        tensorend = torch.cat((tensorend, tensor), dim=0)
        try:
            boardx.pop()
        except IndexError:
            pass
    return tensorend.view(size=(1, 104, 8, 8))

class resblock(nn.Module):
    def __init__(self):
        super(resblock, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x + res
class ChessAttention(nn.Module):
    #not full attention, but only in orthagonal, diagonal, knight moves
    def __init__(self, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        
        
class azt(nn.Module):
    def __init__(self, input_size=104, num_move_categories=1858):
        super(azt, self).__init__()
        self.board_size = 8
        self.conv1 = nn.Conv2d(104, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.resblock = nn.Sequential(*[resblock() for _ in range(12)])
        self.policy_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc1 = nn.Linear(128, 256)
        self.policy_fc2 = nn.Linear(256, num_move_categories)
        self.value_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(2)
        self.value_fc1 = nn.Linear(128, 256)
        self.value_fc2 = nn.Linear(256, 1)
        self.cp_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.cp_bn = nn.BatchNorm2d(2)
        self.cp_fc1 = nn.Linear(128, 256)
        self.cp_fc2 = nn.Linear(256, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        # Policy head
        policy_x = self.policy_conv(x)
        policy_x = self.policy_bn(policy_x)
        policy_x = torch.relu(policy_x)
        policy_x = policy_x.view(policy_x.size(0), -1)
        policy_x = self.policy_fc1(policy_x)
        policy_output = self.policy_fc2(policy_x)
        policy_output = nn.functional.softmax(policy_output, dim=1)
        # Value head
        value_x = self.value_conv(x)
        value_x = self.value_bn(value_x)
        value_x = torch.relu(value_x)
        value_x = value_x.view(value_x.size(0), -1)
        value_x = self.value_fc1(value_x)
        value_x = self.value_fc2(value_x)
        value_output = torch.tanh(value_x)
        cp_x = self.cp_conv(x)
        cp_x = self.cp_bn(cp_x)
        cp_x = cp_x.view(cp_x.size(0), -1)
        cp_x = self.cp_fc1(cp_x)
        cp_x = self.cp_fc2(cp_x)
        return policy_output, value_output, cp_x

model = azt().to(device)
model.eval()

def save1(modelx):
    torch.save(modelx.state_dict(), "chess_model.pth")
    print("Model saved successfully!")

class MCTSWorker(threading.Thread):
    def __init__(self, root, simulations, server, lock, batch_queue, result_queue, worker_id):
        super().__init__()
        self.root = root
        self.simulations = simulations
        self.server = server
        self.lock = lock
        self.batch_queue = batch_queue
        self.result_queue = result_queue
        self.worker_id = worker_id

    def run(self):
        for _ in range(self.simulations):
            node_path = []
            node = self.root
            # Selection
            while node.is_expanded and node.children:
                node = node.choosechild()
                node_path.append(node)
                # Virtual loss
                with self.lock:
                    node.visits += 1
                    node.Q -= 1  # virtual loss

            # Expansion
            if not node.board.is_game_over():
                node.extend()
                for child in node.children:
                    self.batch_queue.put((child, child.tensor))
            else:
                result = node.board.result()
                value = -1 if result == '0-1' else 1 if result == '1-0' else 0
                node.value = value
                node.Q = value

            # Wait for evaluation results
            evaluated = False
            while not evaluated:
                try:
                    eval_node, Ps, Vs, CPs = self.result_queue.get(timeout=1)
                    eval_node.P = Ps
                    eval_node.value = Vs
                    eval_node.Q = Vs
                    evaluated = True
                except queue.Empty:
                    continue

            # Backpropagation with virtual loss reset
            for n in reversed(node_path):
                with self.lock:
                    n.visits -= 1  # remove virtual loss
                    n.update_Q(eval_node.value)

def MCTSsearch(root, simulations, num_workers=4):
    server = get_inference_server()
    lock = threading.Lock()
    batch_queue = queue.Queue()
    result_queue = queue.Queue()
    workers = [MCTSWorker(root, simulations // num_workers, server, lock, batch_queue, result_queue, i)
               for i in range(num_workers)]
    for w in workers:
        w.start()
    running = True
    while running:
        batch = []
        nodes = []
        while not batch_queue.empty() and len(batch) < 32:
            node_item, tensor = batch_queue.get()
            batch.append(tensor.view(104, 8, 8))
            nodes.append(node_item)
        if batch:
            batch_tensor = torch.stack(batch)
            start = time.time()
            Ps_np, Vs_np, CPs_np = server.infer(batch_tensor, timeout=30)
            elapsed = time.time() - start
            npos = batch_tensor.size(0)
            with _stats_lock:
                _eval_stats["batches"] += 1
                _eval_stats["positions"] += int(npos)
                _eval_stats["total_time"] += elapsed
                _eval_stats["batch_times"].append(elapsed)
            for i, node_item in enumerate(nodes):
                # CPs_np exists from server response
                result_queue.put((node_item, Ps_np[i].tolist(), float(Vs_np[i]), CPs_np[i]))
        running = any(w.is_alive() for w in workers)
    for w in workers:
        w.join()
    # build distribution and pick best child
    visits_list = [child.visits for child in root.children]
    best_child = None
    if root.children:
        max_vis = max(visits_list)
        # tie-break by first max
        for child in root.children:
            if child.visits == max_vis:
                best_child = child
                break
    sum_vis = sum(visits_list)
    if sum_vis > 0:
        distribution = [v / sum_vis for v in visits_list]
    else:
        distribution = [1.0 / len(visits_list) for _ in visits_list] if visits_list else []
    return best_child, distribution

def benchmark():
    reset_eval_stats()
    root = node(chess.Board(), crunch_board(chess.Board()), None, None)
    bot = Bot(simulations=100, num_cores=4)
    move, distr = bot.choose_move(root)
    print("Best move:", move)
    stats = get_eval_stats()
    print(f"Eval batches: {stats['batches']}, positions: {stats['positions']}, total_time: {stats['total_time']:.3f}s")
    print(f"Avg batch time: {stats['avg_batch_time']:.4f}s, throughput: {stats['throughput_pos_per_sec']:.1f} pos/s")
    # Optionally print more stats here
    
if __name__ == "__main__":
    try:
        benchmark()
    finally:
        server = get_inference_server()
        server.stop()