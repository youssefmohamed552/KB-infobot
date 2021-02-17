import torch
from torch.distributions import Categorical

class BeliefSummary:
  def __init__(self, i_dim, j_dim, M_sigh):
    self.i_dim = i_dim
    self.j_dim = j_dim
    self.M_sigh = M_sigh
    self.w = []

  def forward(self, pTs, p_0):
    wt = []
    for j in range(self.j_dim):
      pT_v = 0
      pT_sigh = 0
      for i in range(self.i_dim):
        print(pTs[i][0])
        if i in self.M_sigh[j]:
          pT_sigh += pTs[i][0]
        else:
          pT_v += pTs[i][0]

      wt_j = torch.tensor([pT_v + p for p in [pT_sigh * p for p in p_0[j]]])
      wt_j = Categorical(probs=wt_j).entropy()
      print("wt_j : {}".format(wt_j))
      wt.append(wt_j)
    self.w.append(wt)
    return self.w[-1]


if __name__ == '__main__' :
  bs = BeliefSummary(3, 2, [[], [1], [0]])
  w = bs.forward([[0.21], [0.14], [0.64]], [[0.4,0.3], [0.2,0.1], [0.2,0.5]])
  print(w)
