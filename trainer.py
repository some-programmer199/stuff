import chess.engine
import chess.pgn
import torch
import chess
import torch.nn as nn
import torch.optim as optim
import zstandard as zst
import io
import chardet
import random
import subprocess as sub
import Bot
import sys
import matplotlib.pyplot as plt
from collections import deque
import moves
import math
from multiprocessing import Pool
import multiprocessing as mp
import queue  # Add this import at the top
def prep_data():
    board = chess.Board()
    with open('data.pgn.zst', 'rb') as data:
        dctx = zst.ZstdDecompressor()
        reader = dctx.stream_reader(data)
        text = io.TextIOWrapper(reader, encoding='utf-8-sig', errors='replace')
        global boards
        boards = []
        errors = []
        for games in range(100):  # Reduced to 10 games for faster processing
            board = chess.Board()
            game = chess.pgn.read_game(text)
            if game is None:
                break
            for move in game.mainline_moves():
                try:
                    board.push(move)
                    if board.ply() >= 2 and not board.is_game_over():
                        boards.append(board.copy())
                    else:
                        errors.append(board.ply())
                except (AssertionError):
                    break
        
    random.shuffle(boards)
def gen_ranboard(amount):
    selected = []
    for i in range(amount):
        selected.append(random.choice(boards))
    return selected
lc01 = sub.Popen(r"C:\Users\26jxu\desktop\stuff\lc0\lc0.exe",
                    stdin=sub.PIPE,
                    stdout=sub.PIPE,
                    stderr=sub.PIPE,
                    universal_newlines=True,
                    bufsize=1)
def send(cmd):
        lc01.stdin.write(cmd + "\n")
        lc01.stdin.flush()
def communicate_policy(boardx:chess.Board):
    
    board_fen=boardx.fen()
    
    boardx.fen()
    # Initialize UCI protocol
    send("uci")
    while True:
        line = lc01.stdout.readline()
        if "uciok" in line:
            break

    # Set options for policy extraction
    send("setoption name MultiPV value 1")
    send("setoption name VerboseMoveStats value true")
    send("isready")
    while True:
        line = lc01.stdout.readline()
        if "readyok" in line:
            break

    # Set position and start search
    send(f"position fen {board_fen}")
    send("go nodes 1")  # Use minimal search for raw policy
    policy={}
    value={}
    # Read policy from engine output
    while True:
        line=lc01.stdout.readline()
        if line.startswith("info") and "string" in line:
                parts=line.strip().split()
                move=parts[2]
                if move == "node":
                 break
                for i in range(len(parts)):
                    if parts[i]=="(P:":
                        pscore=parts[i+1]
                        
                policy[move]=pscore[:-2]
                
        if line.startswith("bestmove"):
            break
    # Normalize policy to get probabilities

    # Quit engine
    
    """
    242 total moves
    p:8*8
    r:4*7*2
    n:8*2
    b:7*4*2
    k:10
    q:7*8

    """
    
    tens=[0]*1858
    for move in policy:
     if len(move)==4:
        fromsq=move[:2]
        
        start_idx,end_idx = moves.sq_idx[fromsq]
     else:
        start_idx,end_idx = moves.sq_idx["promotion"]
     i=0
     for pmove in moves.pmoves[start_idx:end_idx]:
         if pmove==move:
             tens[start_idx+i]=float(policy[move])/100
             
             break
         i+=1
    total=sum(tens)
    tens = [x / total for x in tens]
    total=sum(tens)
    return tens

def engine_worker(input_queue, output_queue,error_queue, lc0_path, stockfish_path):
    # Start engines once per worker
    lc0 = chess.engine.SimpleEngine.popen_uci(lc0_path)
    lc0.configure({"UCI_ShowWDL": True})
    stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    data=None
    while True:
        item = input_queue.get()
        if item == "STOP":
            break
        board, depth = item
        try:
            limit = chess.engine.Limit(depth=depth)
            datas = stockfish.analyse(board, limit)
            
            datal = lc0.analyse(board, limit)
            data=datal.get('score')
            wdl=datal.get('wdl')
            
            win,draw,loss=wdl.pov(board.turn)
            total=win+draw+loss
            
            scores=datas.get('score').pov(board.turn)
            

            wps= 1 / (1 + math.exp(-scores.score() / 100.0))
            wpl=(win+draw/2)/total
            winprob=wpl*0.9 +wps*0.1
            cp=scores.score()
            output_queue.put((winprob, cp))
            
        except Exception as e:
            output_queue.put((None, None, str(e),data))
    lc0.close()
    stockfish.close()

#winprobs = 1 / (1 + math.exp(-datas / 100.0))
def tester(boards: list, timeout=30):
    for i, board in enumerate(boards):
        input_queues[i % num_workers].put((board, 4))
    results = []
    for i, board in enumerate(boards):
        try:
            result = output_queues[i % num_workers].get(timeout=timeout)
            print(result)
        except queue.Empty:
            print(f"Timeout waiting for engine result for board {i}")
            result = (None, None, "Timeout")
        results.append(result)
    try:
     print(error_queue.get(timeout,10))
    except(queue.Empty):
        pass
    return results
ll = deque(maxlen=50)
def train(model: nn.Module, epochs):
    batchloss=0
    boards = gen_ranboard(epochs*4)
    criterionv = nn.MSELoss()
    criterionp=nn.CrossEntropyLoss()
    criterioncp=nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    plt.ion()
    fig,ax=plt.subplots()
    line1,=ax.plot(ll,marker=",")
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    for epoch in range(0,epochs*4,4):
      try:
        optimizer.zero_grad()
        try:
            batch=boards[epoch:epoch+4]
            results=tester(batch)
            tenslist=torch.stack([Bot.crunch_board(x).view(104,8,8) for x in batch])
            outputps,outputvs,outputcps = model.forward(tenslist.to(Bot.device))
            
        except (TypeError, chess.engine.EngineError) as e:
            print(f"Skipping board due to error: {e}")
            continue  # Skip to the next board
        for i, board in enumerate(batch):
          batchloss=0
          try:
             targetv,targetcp=results[i][:2]
             outputp=outputps[i]
             outputv=outputvs[i]
             outputcp=outputcps[i]
             targetv = torch.tensor(targetv, dtype=torch.float32)
             targetcp=torch.tensor(targetcp,dtype=torch.float32)
             targetp= torch.tensor(communicate_policy(boards[epoch]),dtype=torch.float32)
             
             lossv = criterionv(outputv, targetv)
             losscp=criterioncp(outputcp,targetcp)/100
             lossp= criterionp(outputp,targetp)
             loss=lossp+lossv+losscp
             batchloss+=loss
             
             try:
              ll.append(loss.item())
             except(TypeError):
              ll.append(0.0)
              scheduler.step()
             line1.set_ydata(ll)
             line1.set_xdata(range(len(ll))) 
             ax.autoscale_view()
             ax.relim()
             plt.draw()
             plt.pause(0.01)
             plt.show()
          except (TypeError, chess.engine.EngineError) as e:
            print(f"Skipping board due to error: {e}")
        if batchloss != 0:
         batchloss.backward()
         optimizer.step()
      except(KeyboardInterrupt):
          
          save1(model)
prep_data()


def load_model(modelx, filepath="chess_model.pth"):
    """Loads the model weights from the specified file."""
    try:
        modelx.load_state_dict(torch.load(filepath))
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    except Exception as e:
        print(f"Error loading model: {e}")

# Example usage:
  # Reduced to 10 epochs for faster training
def save1(modelx):
    torch.save(modelx.state_dict(), "chess_model.pth")
    print("Model saved successfully!")
if __name__ == "__main__":
  lc0_path = r"C:\Users\26jxu\desktop\stuff\lc0\lc0.exe"
  stockfish_path = r"C:\Users\26jxu\desktop\stuff\stockfish\stockfish-windows-x86-64-avx2.exe"
  num_workers = 2
  input_queues = [mp.Queue() for _ in range(num_workers)]
  output_queues = [mp.Queue() for _ in range(num_workers)]
  error_queue=mp.Queue()
  workers = []
  for i in range(num_workers):
    p = mp.Process(target=engine_worker, args=(input_queues[i], output_queues[i],error_queue, lc0_path, stockfish_path))
    p.start()
    workers.append(p)
  load_model(Bot.model)
# Move save1 to the end, after training
  pos=gen_ranboard(1)[0]
  save1(Bot.model)
  try:
   train(Bot.model,1000)
  finally:
   for q in input_queues:
        q.put("STOP")
   for p in workers:
        p.join()
   lc01.terminate()