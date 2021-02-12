import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BeliefTracker:
  def __init__(self, input_dim, hidden_dim):
    self.hidden_dim = hidden_dim
    self.gru = nn.GRU(input_dim, hidden_dim).float()

  def forward(self, sentence):
    hidden = self.initHidden()
    for x in sentence:
      embedding = torch.tensor(x).view(1,1,-1)
      output, hidden = self.gru(embedding.float(), hidden.float())
    return output

  def initHidden(self):
    return torch.zeros(1,1, self.hidden_dim, device=device)
   

if __name__ == '__main__':
  print('testing belief tracker')
  inp_dim = 10
  hidden_dim = 16
  encoded_statement = [np.random.rand(inp_dim) for _ in range(5)]
  b = BeliefTracker(inp_dim, hidden_dim)
  out = b.forward(encoded_statement)
  print(out)
