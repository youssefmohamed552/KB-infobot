import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNetwork:
  def __init__(self, input_dim, hidden_dim, i_dim):
    self.hidden_dim = hidden_dim
    self.i_dim = i_dim
    self.gru = nn.GRU(input_dim, hidden_dim)

  def forward(self, state):
    hidden = self.initHidden()
    print('in forward')
    for s in state:
      s = s.view(1,1,-1)
      print("s : {}".format(s))
      output, hidden = self.gru(s, hidden)
    linear = nn.Linear(self.hidden_dim, self.i_dim)
    output = linear(output)
    print('output: {}'.format(output))
    soft = nn.Softmax(dim=0)
    pi = soft(output)
    return pi
    

  def initHidden(self):
    return torch.zeros(1,1, self.hidden_dim, device=device)


if __name__ == '__main__':
  pn = PolicyNetwork(3, 2)
  s1 = torch.tensor([0.4, 0.1, 0.5])
  s2 = torch.tensor([0.4, 0.6, 0.1])
  s = [s1, s2]
  h = pn.forward(s)
  print(h)


