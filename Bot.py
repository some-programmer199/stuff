import chess
import random
import torch
import torch.nn as nn
import torch.optim as optim
import moves
import math
import datetime
from multiprocessing import Pool, Process, Manager
import multiprocessing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Inference server (separate process owns GPU) ----------
def inference_worker(request_queue, device_str):
    # This function runs in a separate process and imports the model classes
    # directly from this module (which is safe because the process is spawned).
    import torch
    from Bot import azt  # import model class from this module
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
            # move results to CPU numpy
            Ps_np = Ps.detach().cpu().numpy()
            Vs_np = Vs.detach().cpu().numpy().reshape(-1)  # shape (N,)
            CPs_np = CPs.detach().cpu().numpy()
        response_queue.put((Ps_np, Vs_np, CPs_np))
    # clean exit
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
            # send sentinel and join
            self.request_queue.put(None)
            self.process.join(timeout=5)
            self.started = False
    def infer(self, batch_tensor: torch.Tensor, timeout=None):
        """
        batch_tensor: torch.Tensor (N,104,8,8) on any device — will be converted to numpy (CPU)
        returns: Ps_np, Vs_np, CPs_np as numpy arrays
        """
        if not self.started:
            self.start()
        batch_numpy = batch_tensor.detach().cpu().numpy()
        response_q = self.manager.Queue()
        self.request_queue.put((batch_numpy, response_q))
        result = response_q.get(timeout=timeout)
        return result

# single global server instance (created lazily)
_inference_server = None
def get_inference_server():
    global _inference_server
    if _inference_server is None:
        _inference_server = InferenceServer(device_str=("cuda" if torch.cuda.is_available() else "cpu"))
        _inference_server.start()
    return _inference_server

# ---------- existing Bot implementation, modified to use inference server ----------
class Bot:
    def __init__(self,simulations=100,num_cores=1):
        self.simulations=simulations
        self.num_cores = num_cores
        # ensure inference server is running when a bot is created
        get_inference_server()
    def choose_move(self,board:chess.Board):
        root=node(board,crunch_board(board),None,None)
        bestmove, distribution=MCTSsearch(root,self.simulations)
        return bestmove.move, distribution

def movetoNN(move:chess.Move):
   move=str(move)
   if len(move)==4:
        fromsq=move[:2]
        
        start_idx,end_idx = moves.sq_idx[fromsq]
   else:
        start_idx,end_idx = moves.sq_idx["promotion"]
   i=0
   for pmove in moves.pmoves[start_idx:end_idx]:
         if pmove==move:
            idx=i+start_idx
         i+=1
   return idx
class node:
   def __init__(self,board:chess.Board,tensor:torch.Tensor,parent,move):
      self.board=board
      self.tensor=tensor
      self.visits=1
      self.move=move
      self.children=[]
      self.is_expanded=False
      self.P,self.value,_=[0]*1858,0,0
      self.Q=self.value
      self.updates=1
   def __repr__(self):
      return str(self.move)
   def extend(self):
       for move in self.board.legal_moves:
           boardx=self.board.copy()
           boardx.push(move)
           self.children.append(node(boardx,self.tenspush(move),self,move))
           self.is_expanded=True
   def choosechild(self,cpuct=1.5):
       best_child=None
       best_value=-math.inf
       for child in self.children:
           Q=child.Q
           prob=self.P[movetoNN(child.move)]
           a=child.visits+1
           U=cpuct*prob*math.sqrt(self.visits)/a
           v=Q+U
           if v >= best_value:
               best_child=child
               best_value=v
       return best_child
   def tenspush(self, move: chess.Move):
       # Clone the tensor to avoid in-place modification
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

       # Handle promotion
       if move.promotion:
           pawn_plane = piece_to_plane[chess.PAWN] + (0 if piece.color == chess.WHITE else 6)
           tensor[0, pawn_plane, fromr, fromc] = 0.0
           promo_plane = piece_to_plane[move.promotion] + (0 if piece.color == chess.WHITE else 6)
           tensor[0, promo_plane, tor, toc] = 1.0
           return tensor

       # Handle castling
       if self.board.is_castling(move):
           king_plane = piece_to_plane[chess.KING] + (0 if piece.color == chess.WHITE else 6)
           tensor[0, king_plane, fromr, fromc] = 0.0
           tensor[0, king_plane, tor, toc] = 1.0
           if toc > fromc:  # kingside
               rook_fromc = 7
               rook_toc = 5
           else:  # queenside
               rook_fromc = 0
               rook_toc = 3
           rook_row = fromr
           rook_plane = piece_to_plane[chess.ROOK] + (0 if piece.color == chess.WHITE else 6)
           tensor[0, rook_plane, rook_row, rook_fromc] = 0.0
           tensor[0, rook_plane, rook_row, rook_toc] = 1.0
           return tensor

       # Handle en passant
       if self.board.is_en_passant(move):
           pawn_plane = piece_to_plane[chess.PAWN] + (0 if piece.color == chess.WHITE else 6)
           tensor[0, pawn_plane, fromr, fromc] = 0.0
           tensor[0, pawn_plane, tor, toc] = 1.0
           cap_row = tor + (1 if piece.color == chess.WHITE else -1)
           tensor[0, pawn_plane, cap_row, toc] = 0.0
           return tensor

       # Normal move
       plane = piece_to_plane[piece.piece_type] + (0 if piece.color == chess.WHITE else 6)
       tensor[0, plane, fromr, fromc] = 0.0
       tensor[0, plane, tor, toc] = 1.0
       return tensor
   def update_Q(self,valuelist:list):
       start=self.Q*self.updates
       newavg=(start+sum(valuelist))/(self.updates+len(valuelist))
       self.visits+=1
       self.updates+=len(valuelist)
timestamps=[]
depths=[]
batch=[]
def MCTSsearch(root:node,sim):
    server = get_inference_server()
    for _ in range(sim):
        n=root
        depth=0
        path=[]
        while n.is_expanded:
            path.append(n)
            n = n.choosechild()
            depth+=1
        batch=[]
        
        n.extend()
        batch=n.children
        # create batched input for inference
        batchx=torch.stack([x.tensor.view(104,8,8) for x in batch])
        # send to inference server (which runs on the GPU)
        Ps_np, Vs_np, CPs_np = server.infer(batchx, timeout=30)  # returns numpy arrays
        # assign outputs to children
        for i, child in enumerate(batch):
                child.P = Ps_np[i].tolist()
                child.value = float(Vs_np[i])
                child.Q = child.value
            
        new_values=[]
        for child in n.children:
            new_values.append(child.value)
        for n in reversed(path):
            n.update_Q(new_values)
        timestamps.append(datetime.datetime.now())
        depths.append(depth)
    best_value=0
    best_child=None
    distribution=[]
    for child in root.children:
        distribution.append(child.visits)
        if child.visits >= best_value:
            best_child=child
            best_value=child.visits
    distribution=[x/sum(distribution) for x in distribution]
    return best_child,distribution
piece_to_plane={
        chess.PAWN:0,
        chess.KNIGHT:1,
        chess.BISHOP:2,
        chess.ROOK:3,
        chess.QUEEN:4,
        chess.KING:5,
        }
def crunch_board(boardx:chess.Board):
    tensorend=torch.zeros((0,8,8),dtype=torch.float32)
    for x in range(8):
      tensor=torch.zeros(13,8,8)
      for square in chess.SQUARES:
        piece=boardx.piece_at(square)
        if piece:
            row=7-chess.square_rank(square)
            col=chess.square_file(square)
            base= 0 if piece.color==chess.WHITE else 6
            plane=base+piece_to_plane[piece.piece_type]
            tensor[plane,row,col]=1.0
      tensor[12,:,:]= float(boardx.ply())
      tensorend=torch.cat((tensorend,tensor),dim=0)
      try:
       boardx.pop()
      except(IndexError):
         pass
    return tensorend.view(size=(1,104,8,8))

class resblock(nn.Module):
    def __init__(self):
        super(resblock,self).__init__()
        self.conv1=nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(256)
        self.conv2=nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.bn2=nn.BatchNorm2d(256)
    def forward(self,x):
        res=x
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        return x+res
# Define a simple neural network
class azt(nn.Module):
    def __init__(self, input_size=104, num_move_categories=1858):
        super(azt, self).__init__()
        self.board_size = 8
        self.conv1 = nn.Conv2d(104, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.bn2= nn.BatchNorm2d(256)
        self.resblock=nn.Sequential(*[resblock() for _ in range(12)])
        self.policy_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc1=nn.Linear(128,256)
        self.policy_fc2 = nn.Linear(256, num_move_categories)
        
        self.value_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(2)
        self.value_fc1=nn.Linear(128,256)
        self.value_fc2 = nn.Linear(256, 1)
        self.cp_conv=nn.Conv2d(256, 2, kernel_size=1)
        self.cp_bn=nn.BatchNorm2d(2)
        self.cp_fc1=nn.Linear(128,256)
        self.cp_fc2=nn.Linear(256, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x = torch.relu(x)
        policy_output = None
        value_output = None

        if True:
            # Policy head
            policy_x = self.policy_conv(x)
            policy_x = self.policy_bn(policy_x)
            policy_x = torch.relu(policy_x)
            policy_x = policy_x.view(policy_x.size(0),-1)
            policy_x=self.policy_fc1(policy_x)
            policy_output = self.policy_fc2(policy_x)
            policy_output = nn.functional.softmax(policy_output, dim=1)
            # Value head
            value_x = self.value_conv(x)
            value_x = self.value_bn(value_x)
            value_x = torch.relu(value_x)
            value_x = value_x.view(value_x.size(0),-1)
            value_x = self.value_fc1(value_x)
            value_x=self.value_fc2(value_x)
            value_output=torch.tanh(value_x)

            cp_x=self.cp_conv(x)
            cp_x=self.cp_bn(cp_x)
            cp_x=cp_x.view(cp_x.size(0), -1)
            cp_x=self.cp_fc1(cp_x)
            cp_x=self.cp_fc2(cp_x)
            return policy_output,value_output,cp_x
# Initialize the model
model = azt().to(device)
x=crunch_board(chess.Board())

model.eval()
# Save the model's state dictionary
def save1(modelx):
  torch.save(modelx.state_dict(), "chess_model.pth")
  print("Model saved successfully!")
def benchmark():
    root=node(chess.Board(),crunch_board(chess.Board()),None,None)
    MCTSsearch(root,500)
    speed=[]

    datetime.timedelta().total_seconds
    for i,time in enumerate(timestamps):
     try:
      prev=timestamps[i-1]
      speed.append((time-prev).total_seconds())
     except IndexError:
        pass
    for i in range(len(speed)):
     print(f"sim {i+1}, speed {speed[i]}, depth {depths[i]}")
if __name__ == "__main__":
    benchmark()
