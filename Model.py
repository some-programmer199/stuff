# X.py
import os
import time
import random
import pickle
import chess
import torch
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import lmdb
from typing import Optional, List, Dict, Any
from Model import ChessAttention, evaluator, MAX_PIECES, CONTEXT_LENGTH

os.makedirs('./lmdb_data', exist_ok=True)
env = lmdb.open('./lmdb_data', map_size=2**40, writemap=True)

class Node:
    def __init__(self, board: chess.Board, context: Optional[torch.Tensor] = None):
        self.board = board
        self.board_tensor = self._to_tensor()
        if context is None:
            context = torch.zeros(CONTEXT_LENGTH)
        self.context = context.clone().detach() if isinstance(context, torch.Tensor) else torch.from_numpy(context).float()

    def _to_tensor(self):
        t = torch.zeros((MAX_PIECES, 4), dtype=torch.float32)
        for i, (sq, p) in enumerate(board.piece_map().items()):
            if i >= MAX_PIECES - 1: break
            t[i, 0] = p.piece_type - 1
            t[i, 1] = int(p.color)
            t[i, 2] = (sq // 8) - 3.5
            t[i, 3] = (sq % 8) - 3.5
        t[-1, 0] = 10.0
        t[-1, 1] = board.ply()
        return t

    def to(self, device):
        self.board_tensor = self.board_tensor.to(device)
        self.context = self.context.to(device)
        return self


class MCTSNode:
    _id_counter = mp.Value('i', 0)
    def __init__(self, state: Node, parent, prior: float, move_uci: Optional[str], turn: chess.Color):
        with MCTSNode._id_counter.get_lock():
            self.id = MCTSNode._id_counter.value
            MCTSNode._id_counter.value += 1
        self.state = state
        self.parent = parent
        self.move = move_uci
        self.turn = turn
        self.prior = prior
        self.children: Dict[str, int] = {}
        self.N = self.W = self.W_anti = 0
        self.Q = self.antiQ = self.variance = 0.0
        self.is_expanded = False
        self._save()

    def _save(self):
        d = {
            'id': self.id, 'fen': self.state.board.fen(), 'move': self.move,
            'N': self.N, 'W': self.W, 'W_anti': self.W_anti, 'Q': self.Q, 'antiQ': self.antiQ,
            'variance': self.variance, 'is_expanded': self.is_expanded,
            'children': self.children.copy(), 'turn': int(self.turn),
            'prior': self.prior, 'context': pickle.dumps(self.state.context.cpu())
        }
        with env.begin(write=True) as txn:
            txn.put(f'node_{self.id}'.encode(), pickle.dumps(d))


class MCTS:
    def __init__(self, model: ChessAttention, num_workers=4, batch_size=32):
        self.model = model.eval()
        self.model.share_memory()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.eval_queue = mp.Queue(maxsize=1024)
        self.shutdown = mp.Event()
        self.workers: List[mp.Process] = []

    def _worker(self, root_id: int):
        while not self.shutdown.is_set():
            try:
                with env.begin() as txn:
                    root_data = txn.get(f'node_{root_id}'.encode())
                    if not root_data: continue
                    node = pickle.loads(root_data)
                    path = [node['id']]
                    while node['is_expanded'] and node['children']:
                        # simple max-N selection for speed in worker
                        child_id = max(node['children'].values(), key=lambda cid: pickle.loads(txn.get(f'node_{cid}'.encode()) or b'')get('N',0))
                        node = pickle.loads(txn.get(f'node_{child_id}'.encode()))
                        path.append(child_id)
                    self.eval_queue.put((node['id'], node['fen'], node['context'], path))
            except:
                time.sleep(0.01)

    def search(self, board: chess.Board, sims: int = 1600, context=None):
        root_state = Node(board, context)
        root = MCTSNode(root_state, None, 1.0, None, board.turn)
        self._expand_root(root)

        for _ in range(self.num_workers):
            p = mp.Process(target=self._worker, args=(root.id,), daemon=True)
            p.start()
            self.workers.append(p)

        pbar = tqdm(total=sims, desc="MCTS (batched)", colour="cyan")
        done = 0
        while done < sims:
            batch = []
            while len(batch) < self.batch_size and done < sims:
                try:
                    batch.append(self.eval_queue.get(timeout=0.5))
                    done += 1
                except:
                    break

            if not batch: continue

            nodes = []
            for _, fen, ctx_pickled, _ in batch:
                board = chess.Board(fen)
                ctx = pickle.loads(ctx_pickled) if isinstance(ctx_pickled, (bytes, bytearray)) else ctx_pickled
                node = Node(board, ctx).to('cuda' if torch.cuda.is_available() else 'cpu')
                nodes.append(node)

            # REAL BATCH EVAL
            x = torch.stack([n.board_tensor for n in nodes])
            c = torch.stack([n.context for n in nodes])
            with torch.no_grad():
                v, av, var, _, pol, new_ctx = self.model(x, c)

            # Backup each
            for i, (leaf_id, _, _, path) in enumerate(batch):
                self._backup(
                    leaf_id=leaf_id,
                    path=path,
                    value=v[i].item(),
                    antivalue=av[i].item(),
                    variance=var[i].item(),
                    new_context=new_ctx[i].cpu(),
                    policy_logits=pol[i].cpu()
                )
            pbar.update(len(batch))

        self.shutdown.set()
        for p in self.workers: p.join(1)
        pbar.close()

        # Pick best move
        with env.begin() as txn:
            root_data = pickle.loads(txn.get(f'node_{root.id}'.encode()))
            visits = {}
            for uci, cid in root_data['children'].items():
                child = pickle.loads(txn.get(f'node_{cid}'.encode()))
                visits[uci] = child['N']
        best = max(visits, key=visits.get)
        return best, visits

    def _expand_root(self, root: MCTSNode):
        policy, v, av, var, _, ctx = evaluator(root.state, self.model)
        board = root.state.board
        for move in board.legal_moves:
            uci = move.uci()
            board.push(move)
            child = MCTSNode(Node(board, ctx), root, policy.get(uci, 0.0), uci, not board.turn)
            root.children[uci] = child.id
            board.pop()
        root.is_expanded = True
        root.N = 1; root.W = v; root.W_anti = av; root.Q = v; root.antiQ = av; root.variance = var
        root._save()

    def _backup(self, leaf_id, path, value, antivalue, variance, new_context, policy_logits):
        # simplified backup - just update N/W/variance
        for nid in reversed(path):
            with env.begin(write=True) as txn:
                data = pickle.loads(txn.get(f'node_{nid}'.encode()))
                data['N'] += 1
                data['W'] += value
                data['W_anti'] += antivalue
                data['Q'] = data['W'] / data['N']
                data['antiQ'] = data['W_anti'] / data['N']
                if nid == leaf_id:
                    data['variance'] = variance
                    data['is_expanded'] = True
                    data['context'] = pickle.dumps(new_context)
                txn.put(f'node_{nid}'.encode(), pickle.dumps(data))


if __name__ == "__main__":
    model = ChessAttention()
    if torch.cuda.is_available():
        model = model.cuda()

    mcts = MCTS(model, num_workers=6, batch_size=48)
    board = chess.Board()
    best_move, visits = mcts.search(board, sims=1024)
    print(f"\nBest move: {best_move}")
    print("Top moves:", sorted(visits.items(), key=lambda x: -x[1])[:8])
