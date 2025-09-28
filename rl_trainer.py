import torch
import Bot
import chess
from torch.optim import lr_scheduler as lrs
def add_bar():
    print('-',end='')
def play_game(bot1:Bot.Bot,bot2:Bot.Bot):
    board=chess.Board()
    game_states=[]
    max_moves = 400            # safety cap to avoid infinite runs
    move_count = 0
    while not board.is_game_over() and move_count < max_moves:
        if board.turn == chess.WHITE:
            print(f"\nMove {move_count+1}: White thinking...", flush=True)
            move,distr=bot1.choose_move(board)
        else:
            print(f"\nMove {move_count+1}: Black thinking...", flush=True)
            move,distr=bot2.choose_move(board)
        game_states.append((board.copy(),move,distr,0))
        add_bar()
        board.push(move)
        move_count += 1
    if move_count >= max_moves:
        print("\nReached max_moves limit — stopping early.", flush=True)
    result=board.result()
    result=-1 if result=='0-1' else 1 if result=='1-0' else 0
    game_states = [(state, move, distr, result) for state, move, distr, _ in game_states]
    return game_states
def train(optimize:torch.optim.Adam,lrs:lrs.LRScheduler, game_states):
    optimizer.zero_grad()
    total_loss = 0
    for state, move, distr, reward in game_states:
        state_tensor = Bot.crunch_board(state).unsqueeze(0)
        policy, value = Bot.model(state_tensor)
        value_loss = (value.squeeze() - reward) ** 2
        move_index = Bot.move_to_index(move)
        policy_loss = -torch.log(policy.squeeze()[move_index]) * reward
        loss = value_loss + policy_loss
        total_loss += loss
    total_loss.backward()
    optimizer.step()
    lrs.step()
    return total_loss.item() / len(game_states)
if __name__ == "__main__":
    optimizer = torch.optim.Adam(Bot.model.parameters(), lr=0.001)

    lrd=lrs.StepLR(optimizer, step_size=1000, gamma=0.1)
    for epoch in range(10000):
        print(f"\n=== Epoch {epoch+1} ===", flush=True)
        bot_white = Bot.Bot(simulations=50)  # Lower simulations while debugging so choose_move runs quickly
        bot_black = Bot.Bot(simulations=50)  # Lower simulations while debugging so choose_move runs quickly
        game_states = play_game(bot_white, bot_black)
        loss = train(optimizer, lrd, game_states)
        print(f"\nEpoch {epoch+1} completed. Training loss: {loss:.4f}", flush=True)
        if (epoch + 1) % 100 == 0:
            torch.save(Bot.model.state_dict(), f"bot_model_epoch_{epoch+1}.pth")
            print(f"Model saved at epoch {epoch+1}", flush=True)
    # Lower simulations while debugging so choose_move runs quickly
    
    
  