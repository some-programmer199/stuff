import search
import chess
class Bot:
    def __init__(self, searche:search.MCTS name="MCTS Bot",root:search.MCTSNode):
        self.search = searche
        self.name = name
        self.root=root
    def select_move(self, state):
        return self.search.search(state)
    def push(self, move:chess.Move):
        if self.search.root is not None:
            self.search.root = self.search.root.children.get(move.uci(), None)
    

    
    