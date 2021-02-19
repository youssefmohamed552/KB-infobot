import torch

class ActionSelection:
  def __init__(self):
    self.I = []
    self.r = []
  
  def form_I(self, R, pT):
    pT_list = [[i, x] for i, x in enumerate(pT.tolist())]
    pT_list.sort(key=lambda x: x[1])
    pT_list.reverse()

    self.I = [pT_list[i][0] for i in range(R)]

  def reward(self, pT):
    if len(pT) < len(self.I):
      raise Exception('not enough elements in KB for the size of I')
    self.r = torch.tensor([pT[i] for i in self.I])

  def get_mean(self, pT):
    if len(self.r) == 0:
      raise Exception('Either reward() wasn\'t called or the KB is empty')
    mean_I = self.r[0]
    for i in range(1, len(self.I)):
      mean_I *= (self.r[i-1] / (1 - self.r[i]))
    return mean_I
