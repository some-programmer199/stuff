import torch
import Bot
import chess
def add_bar():
    print('-',end='')
def play_game(bot1:Bot.Bot,bot2:Bot.Bot):
    board=chess.Board()
    game_states=[]
    max_moves = 400            # safety cap to avoid infinite runs
    move_count = 0
    while not board.is_game_over() and move_count < max_moves:
        if board.turn:
            print(f"\nMove {move_count+1}: White thinking...", flush=True)
            move=bot1.choose_move(board)
        else:
            print(f"\nMove {move_count+1}: Black thinking...", flush=True)
            move=bot2.choose_move(board)
        game_states.append((board.copy(),move))
        add_bar()
        board.push(move)
        move_count += 1
    if move_count >= max_moves:
        print("\nReached max_moves limit — stopping early.", flush=True)
    result=board.result()
    if result=="1-0":
        reward=1
    elif result=="0-1":
        reward=-1
    else:
        reward=0
    return game_states,reward
if __name__ == "__main__":
    # Lower simulations while debugging so choose_move runs quickly
    bot1=Bot.Bot(simulations=20,num_cores=4)
    bot2=Bot.Bot(simulations=20,num_cores=4)
    game_states,reward=play_game(bot1,bot2)
    for state,move in game_states:
        print(state)
        print(move)
    print("Game result reward:",reward)