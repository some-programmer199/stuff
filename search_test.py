# Prototype: Best-first probabilistic minimax with confidence (python-chess placeholder nets)
# This will try to run a demo using python-chess if available.
# If python-chess is not installed in this environment, the script will print the full code
# so you can copy-run it locally where python-chess exists.
#
# The neural nets are placeholders (fast_eval and full_eval). They use lightweight heuristics:
# - V: normalized material score (from perspective of the side to move)
# - P: softmax over legal moves scored by a simple heuristic (captures prioritized)
# - C: confidence derived from policy entropy (low entropy -> high confidence)
#
# The search is best-first using a frontier (heapq). It expands nodes according to a priority
# combining path_prob, confidence, and surprise (|V - parent_V|).
#
# Run-time knobs near the top of the file for convenience.
#
# This is a prototype for development and experimentation, not a production engine.
import math, random, heapq, sys, textwrap, time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---- Config ----
BUDGET_ITERATIONS = 400      # number of node expansions / attempts
ALPHA = 0.8                  # confidence exponent in priority
BETA = 1.2                   # surprise exponent in priority
EPS_PRUNE = 1e-4             # prune threshold
FULL_EVAL_PROB = 0.15        # chance to run a "full" (slower) eval on expansion
RNG_SEED = 42

random.seed(RNG_SEED)

# ---- Try to import python-chess ----
try:
    import chess
    HAS_PYCHESS = True
except Exception as e:
    HAS_PYCHESS = False

# ---- Helper eval utilities (used whether python-chess present or not) ----
PIECE_VALUES = {
    'P': 1.0, 'N': 3.0, 'B': 3.1, 'R': 5.0, 'Q': 9.0, 'K': 0.0,
    'p': -1.0, 'n': -3.0, 'b': -3.1, 'r': -5.0, 'q': -9.0, 'k': 0.0
}

def material_score_from_fen(fen: str,add_noise=True) -> float:
    # quick material from FEN (works even if python-chess missing)
    board_part = fen.split()[0]
    score = 0.0
    for ch in board_part:
        if ch.isalpha():
            score += PIECE_VALUES.get(ch, 0.0)
    if add_noise:
        score += (random.random() - 0.5) * 0.1  # small noise
    return score

def softmax(vals):
    m = max(vals)
    ex = [math.exp(v - m) for v in vals]
    s = sum(ex)
    if s == 0:
        return [1.0 / len(vals)] * len(vals)
    return [e / s for e in ex]

def entropy(probs):
    return -sum(p * math.log(p + 1e-12) for p in probs)

# ---- Placeholder "Net" ----
class PlaceholderNet:
    """
    fast_eval(state) -> (V, move_list, C)
      V: scalar value in [-1, +1] from side-to-move perspective
      move_list: list of (move, p) where move is a python-chess Move if available, else a string
      C: confidence in [0,1]
    full_eval(state) -> same but more 'accurate' (slightly adjusted)
    """
    def __init__(self):
        pass

    def _score_move(self, board, move) -> float:
        # small heuristic: captures are better, promotions best, otherwise random-ish
        score = 0.0
        if HAS_PYCHESS:
            if board.is_capture(move):
                score += 1.0
            if board.is_checkmate() or board.is_check():
                score += 0.5
            # prefer center-ish moves roughly
            to_sq = move.to_square
            file = chess.square_file(to_sq)
            rank = chess.square_rank(to_sq)
            center_pref = 1.0 - (abs(file - 3.5) + abs(rank - 3.5)) / 7.0
            score += 0.2 * center_pref
        else:
            # fallback, moves are strings - random
            score += random.random() * 0.2
        return score + random.random() * 0.1

    def fast_eval(self, board) -> Tuple[float, List[Tuple[object, float]], float]:
        # V: normalized material in [-1,1] (positive good for side-to-move)
        if HAS_PYCHESS:
            fen = board.fen()
        else:
            fen = board  # in fallback, board is FEN-like string
        m = material_score_from_fen(fen)
        # normalize: assume typical material range within [-39, 39], but scale softer
        V = max(-1.0, min(1.0, 0.03 * m))

        # build policy over legal moves
        moves = []
        if HAS_PYCHESS:
            legal = list(board.legal_moves)
            if not legal:
                move_list = []
            else:
                scores = [self._score_move(board, mv) for mv in legal]
                probs = softmax(scores)
                move_list = list(zip(legal, probs))
        else:
            # fallback: produce a few dummy pseudo-moves
            move_list = [(f"m{i}", 1.0/4) for i in range(4)]
        # Confidence: derived from policy entropy (low entropy -> high confidence)
        probs_only = [p for _, p in move_list] if move_list else [1.0]
        ent = entropy(probs_only)
        # Max entropy for n moves is log(n), scale to [0,1] inverted
        max_ent = math.log(len(probs_only)) if probs_only else 1.0
        C = 1.0 - (ent / (max_ent + 1e-12))  # high entropy -> low confidence
        # add small random jitter so confidence isn't pathological
        C = max(0.01, min(0.99, C * 0.9 + 0.1 * random.random()))
        return V, move_list, C

    def full_eval(self, board) -> Tuple[float, List[Tuple[object, float]], float]:
        # simulate a slightly better evaluation: nudge V toward material but reduce noise
        V, move_list, C = self.fast_eval(board)
        V = max(-1.0, min(1.0, V + 0.05 * (random.random() - 0.5)))
        C = min(1.0, C + 0.1)  # full eval is slightly more confident
        # slightly sharpen policy
        if move_list:
            moves, probs = zip(*move_list)
            probs = list(probs)
            probs = softmax([p * 1.2 for p in probs])
            move_list = list(zip(moves, probs))
        return V, move_list, C

# ---- Search node and functions ----
@dataclass(order=False)
class Node:
    state: object                       # chess.Board or FEN
    parent: Optional['Node'] = None
    move_from_parent: Optional[object] = None
    P_prior: float = 1.0
    children: List['Node'] = field(default_factory=list)
    V: float = 0.0
    C: float = 0.0
    expanded: bool = False
    path_prob: float = 1.0
    depth: int = 0

    def __post_init__(self):
        if self.parent:
            self.path_prob = self.parent.path_prob * self.P_prior
            self.depth = self.parent.depth + 1
        else:
            self.path_prob = 1.0
            self.depth = 0

def priority(node: Node, alpha=ALPHA, beta=BETA, tau=1e-4) -> float:
    parent_V = node.parent.V if node.parent else 0.0
    surprise = abs(node.V - parent_V) + tau
    # higher path_prob -> more important
    return node.path_prob * (max(1e-6, node.C) ** alpha) * (surprise ** beta)

def should_prune(node: Node) -> bool:
    # small-contrib pruning
    return node.path_prob * node.C * abs(node.V) < EPS_PRUNE

def backpropagate(node: Node):
    # recompute node.V and node.C from children, then propagate to parent
    if not node.children:
        return node.V, node.C
    # determine if this is our turn or opponent's by depth parity (0 is root, assume root side to move)
    our_turn = (node.depth % 2 == 0)
    child_values = []
    child_confidences = []
    child_priors = []
    for ch in node.children:
        child_values.append(ch.V)
        child_confidences.append(ch.C)
        child_priors.append(ch.P_prior)
    # if it's our turn, we choose the move that maximizes expected value over opponent replies
    if our_turn:
        # for each child (our move), estimate the opponent response expectation from child's children
        # if child has no children, use the child's V directly (policy assumes opponent distribution later)
        scores = []
        for ch in node.children:
            if not ch.children:
                # opponent not expanded -> fallback to child's V (we'll rely on child's confidence)
                scores.append(ch.P_prior * ch.C * ch.V)
            else:
                # child is an internal node: opponent to move -> expected sum
                opp_sum = 0.0
                for ch2 in ch.children:
                    opp_sum += ch2.P_prior * ch2.C * ch2.V
                scores.append(opp_sum)
        node.V = max(scores) if scores else 0.0
    else:
        # opponent's turn: treat as probabilistic expectation
        s = 0.0
        for ch in node.children:
            s += ch.P_prior * ch.C * ch.V
        node.V = s
    # aggregate confidence as soft OR
    prod = 1.0
    for c in child_confidences:
        prod *= (1.0 - c)
    node.C = max(0.01, min(0.9999, 1.0 - prod))
    # propagate upward
    if node.parent:
        return backpropagate(node.parent)
    return node.V, node.C

def expand_node(node: Node, net: PlaceholderNet):
    # evaluate node quickly, then create children with fast evals
    V0, move_list, C0 = net.fast_eval(node.state)
    node.V = V0
    node.C = C0
    node.expanded = True
    node.children = []
    # create children; move_list is list of (move, p)
    for mv, p in move_list:
        # generate child state
        child_board = node.state.copy()
        child_board.push(mv)
        child_state = child_board
        child = Node(state=child_state, parent=node, move_from_parent=mv, P_prior=p)
        # initialize child's V/C with fast eval (call net.fast_eval)
        cV, c_moves, cC = net.fast_eval(child_state)
        child.V = cV
        child.C = cC
        node.children.append(child)
    # possibility: run full eval on a subset of children
    for ch in list(node.children):
        if random.random() < FULL_EVAL_PROB:
            cV, c_moves, cC = net.full_eval(ch.state)
            ch.V, ch.C = cV, cC

# ---- Selection and deepening policy ----
def select_child_to_deepen(node: Node) -> Optional[Node]:
    # pick child with highest priority (surprise * path_prob * confidence)
    if not node.children:
        return None
    best = max(node.children, key=lambda c: priority(c))
    return best

def deepen_evaluate(child: Node, net: PlaceholderNet):
    # run a full eval (simulate slower estimate)
    cV, c_moves, cC = net.full_eval(child.state)
    child.V, child.C = cV, cC
    # optionally expand the child further (simulate opponent replies)
    # create child's children with fast eval of opponent replies
    if not child.children:
        for mv, p in c_moves[:6]:  # cap branching for demo
            if HAS_PYCHESS:
                new_board = child.state.copy()
                new_board.push(mv)
                new_state = new_board
            else:
                new_state = f"{child.state} {mv}"
            g = Node(state=new_state, parent=child, move_from_parent=mv, P_prior=p)
            gV, g_moves, gC = net.fast_eval(new_state)
            g.V, g.C = gV, gC
            child.children.append(g)

# ---- Top-level best-first search ----
def best_first_search(root_board, net: PlaceholderNet, budget=BUDGET_ITERATIONS):
    root = Node(state=root_board)
    # initial evaluate root
    rV, r_moves, rC = net.fast_eval(root.state)
    root.V, root.C = rV, rC
    # create pseudo-children for root so frontier has something
    root.children = []
    for mv, p in r_moves[:20]:  # limit branching initially
        if HAS_PYCHESS:
            nb = root.state.copy()
            nb.push(mv)
            st = nb
        else:
            st = f"{root.state} {mv}"
        ch = Node(state=st, parent=root, move_from_parent=mv, P_prior=p)
        cV, c_moves, cC = net.fast_eval(st)
        ch.V, ch.C = cV, cC
        root.children.append(ch)
    # initial backprop
    backpropagate(root)

    frontier = []
    # push root children into frontier
    for ch in root.children:
        heapq.heappush(frontier, (-priority(ch), time.time() + random.random(), ch))

    it = 0
    while it < budget and frontier:
        print(it, len(frontier), f"Root V={root.V:.3f} C={root.C:.3f}")
        it += 1
        _, _, node = heapq.heappop(frontier)
        if should_prune(node):
            continue
        # if node not expanded, expand it
        if not node.expanded:
            expand_node(node, net)
            # after expansion, backpropagate
            backpropagate(node)
            # push its children into frontier
            for ch in node.children:
                heapq.heappush(frontier, (-priority(ch), time.time() + random.random(), ch))
            # also reinsert parent for reconsideration
            heapq.heappush(frontier, (-priority(node.parent) if node.parent else -priority(node), time.time()+random.random(), node.parent or node))
        else:
            # if expanded, pick a child to deepen and run a full eval
            ch = select_child_to_deepen(node)
            if ch:
                deepen_evaluate(ch, net)
                backpropagate(ch)
                # push grandchildren
                for g in ch.children:
                    heapq.heappush(frontier, (-priority(g), time.time()+random.random(), g))
        # occasionally re-add a random frontier node to simulate reinspection
        if frontier and random.random() < 0.02:
            i = random.randint(0, len(frontier)-1)
            heapq.heappush(frontier, frontier[i])
    # choose best move from root using V and C and path_prob
    best_child = max(root.children, key=lambda c: (c.C * abs(c.V) * c.path_prob))
    return root, best_child, it

# ---- Demo run ----
def demo():
    net = PlaceholderNet()
    if HAS_PYCHESS:
        board = chess.Board()  # starting position
        print("Running best-first prototype on starting position (python-chess available).")
        root, best_child, iters = best_first_search(board, net, budget=10e100)
        print(f"Finished {iters} iterations. Root V={root.V:.3f}, C={root.C:.3f}.")
        print("Top candidate moves from root (showing up to 8):")
        for ch in sorted(root.children, key=lambda x: -priority(x))[:8]:
            mv = ch.move_from_parent
            print(f"  {mv}  |  V={ch.V:.3f}  C={ch.C:.3f}  path_prob={ch.path_prob:.3f}  priority={priority(ch):.6f}")
        print("\nSelected best child move:", best_child.move_from_parent)
    else:
        print("python-chess not available in this environment. Below is the full implementation code")
        print("you can run locally where python-chess is installed. The code above is self-contained;")
        print("save it to a file and run `python file.py` after `pip install python-chess`.\n")
        # print a small example starter FEN and demo of what would happen
        print(textwrap.dedent("""
        Example usage locally:
          >>> from this_script import best_first_search, PlaceholderNet
          >>> net = PlaceholderNet()
          >>> import chess
          >>> board = chess.Board()
          >>> root, best_child, iters = best_first_search(board, net, budget=400)
          >>> print(best_child.move_from_parent)
        """))

# Run demo
if __name__ == "__main__":
    demo()


