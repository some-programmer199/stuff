import chess
import search
import search_tpu


class SearchConfig:
    def __init__(self, board, num_sims=100, num_workers=4, eps=0.25, alpha=0.03, use_anti=True, device="gpu"):
        self.board = board
        self.num_sims = num_sims
        self.num_workers = num_workers
        self.eps = eps
        self.alpha = alpha
        self.use_anti = use_anti
        self.device = device

    def config(self):
        return self.board, self.num_sims, self.num_workers, self.eps, self.alpha, self.use_anti, self.device


class GameBot:
    def __init__(self, config: SearchConfig, device=None):
        self.config = config
        selected = (device if device is not None else config.device).strip().lower()
        if selected == "tpu":
            self.search_function = search_tpu.run_mcts_parallel
        elif selected in ("gpu", "cpu"):
            self.search_function = search.run_mcts_parallel
        else:
            raise ValueError(f"Unsupported device: {selected}. Use cpu, gpu, or tpu.")

    def get_move(self):
        board, num_sims, num_workers, eps, alpha, use_anti, _ = self.config.config()
        move = self.search_function(
            board,
            sims=num_sims,
            num_workers=num_workers,
            eps=eps,
            alpha=alpha,
            use_anti=use_anti,
        )
        return move

    def push_move(self, move):
        self.config.board.push(move)
