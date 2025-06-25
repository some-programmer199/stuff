import chess
import torch
tensor1=torch.rand(3,10)
tensor2=tensor1[1:]
print((tensor1.shape,tensor2.shape))