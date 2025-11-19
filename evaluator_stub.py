import chess
import moves

def evaluator(node):
    """Simple picklable evaluator used for smoke tests.
    Returns (value, antivalue, raw_policy, variance, antivariance, ctx)
    where raw_policy is a dict mapping legal move UCIs to probabilities.
    """
    board = getattr(node, 'board', None)
    if board is None:
        # try to reconstruct from node.board_tensor -> skip, return default
        return 0.0, 0.0, {}, 1.0, 1.0, getattr(node, 'context', None)
    legal = list(board.legal_moves)
    if not legal:
        return 0.0, 0.0, {}, 0.0, 0.0, getattr(node, 'context', None)
    pol = {m.uci(): 1.0 / len(legal) for m in legal}
    value = 0.0
    antivalue = -value
    variance = 1.0
    antivariance = 1.0
    return value, antivalue, pol, variance, antivariance, getattr(node, 'context', None)
