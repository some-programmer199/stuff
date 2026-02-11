import search
import chess
class SearchConfig:
    def __init__(self, board, num_sims=100, num_workers=4, eps=0.25, alpha=0.03, use_anti=True):
        self.board = board
        self.num_sims = num_sims
        self.num_workers = num_workers
        self.eps = eps
        self.alpha = alpha
        self.use_anti = use_anti
    def config(self):
        return self.board, self.num_sims, self.num_workers, self.eps, self.alpha,self.use_anti
class GameBot:
    def __init__(self, config: SearchConfig, search_function=search.run_mcts_parallel):
        self.search_function = search_function
        self.config = config
    def get_move(self):
        board, num_sims, num_workers, eps, alpha, use_anti = self.config.config()
        move = self.search_function(board, num_sims=num_sims, num_workers=num_workers, eps=eps, alpha=alpha, use_anti=use_anti)
        return move
    def push_move(self, move):
        self.config.board.push(move)