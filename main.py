import chess.svg
import pygame as py
import chess
import sys
import torch
import Bot
import io
py.display.init()
py.display
screen=py.display.set_mode((540,555),py.SCALED)
squarex=540/8
squarey=555/8
p=py.image.load('bp.png')
r=py.image.load('br.png')
n=py.image.load('bn.png')
b=py.image.load('bb.png')
k=py.image.load('bk.png')
q=py.image.load('bq.png')
P=py.image.load('wp.png')
R=py.image.load('wr.png')
N=py.image.load('wn.png')
B=py.image.load('wb.png')
K=py.image.load('wk.png')
Q=py.image.load('wq.png')

py.init
background=py.image.load('board.png').convert()
board=chess.Board()

running=True
print(chess.FILE_NAMES)
print(chess.RANK_NAMES)
def convert(row,c):
    names=chess.SQUARE_NAMES
    return names[(7-row)*8+c]
piece_images = {
        'p': p, 'r': r, 'n': n, 'b': b, 'k': k, 'q': q,  # Black pieces
        'P': P, 'R': R, 'N': N, 'B': B, 'K': K, 'Q': Q   # White pieces
    }
def print_board(board):
    screen.blit(background,(0,0))
    for row in range(8):
        for col in range(8):
            square = chess.square(col, 7 - row)  # Convert to chess square
            piece = board.piece_at(square)
            if piece:
                piece_image = piece_images[piece.symbol()]
                screen.blit(piece_image, (col * squarex, row * squarey))
def nprint_board(boardx:chess.Board):
  bo= chess.svg.board(
    boardx,
    squares=chess.SquareSet(chess.BB_DARK_SQUARES & chess.BB_FILE_B),
    size=400)
  bytes=bo.encode('utf-8')
  surface=py.image.load(io.BytesIO(bytes))

  screen.blit(surface,(0,0))
print_board(board)
data_list=[]

pressed=0
while running:
    if board.turn==chess.WHITE:
       for event in py.event.get():
        if event.type == py.QUIT:
            running = False
            py.quit()
            sys.exit()
        elif event.type == py.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = py.mouse.get_pos()
            col = int(mouse_x / squarex)
            row = int(mouse_y / squarey)
            square_name = chess.square(col,7-row)
            if pressed ==0:
              piece=board.piece_at(square_name)
              if  piece is not None and piece.symbol().isupper() and board.turn==chess.WHITE:
                  pressed+=1
                  piece_selected=piece
                  square_selected=square_name
                  print(piece_selected)
            elif pressed==1:
                move=chess.Move(square_selected,square_name)
                if board.is_legal(move):
                    board.push(move)
                pressed=0
                print_board(board)
    else:
      move= Bot.decision(board)
      board.push(move)
      print_board(board)
    py.display.flip()
