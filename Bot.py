import chess
import random
import torch
import torch.nn as nn
import torch.optim as optim
import moves
import math
import datetime
from multiprocessing import Pool
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def decision(board:chess.Board):
    legal_moves=list(board.legal_moves)
    return random.choice(legal_moves)
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
        batchx=torch.stack([x.tensor.view(104,8,8) for x in batch])
        Ps,Vs,CPs=model.forward(batchx.to(device))
        for i, node in enumerate(batch):
                node.P = Ps[i].tolist()
                node.value = Vs[i].item()
                node.Q = node.value
            
        new_values=[]
        for child in n.children:
            new_values.append(child.value)
        for n in reversed(path):
            n.update_Q(new_values)
        timestamps.append(datetime.datetime.now())
        depths.append(depth)
    best_value=0
    best_child=None
    for child in root.children:
        if child.visits >= best_value:
            best_child=child
            best_value=child.visits
    return best_child
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
