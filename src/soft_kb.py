class SoftKB:
  def __init__(self):
    self.file_name = 'ec-kb.txt'
    self.Nv = []
    self.M_sigh = []
    self.init_table()
    self.j_dim = len(self.relations)
    self.i_dim = len(self.heads)
 
  def init_table(self):
    self.type = None
    self.relations = []
    self.heads = []
    self.tails = []
    with open(self.file_name) as fp:
      line = fp.readline().strip().split('|')
      self.type = line[0]
      self.relations = line[1:]
      for line in fp:
        els = line.strip().split('|')
        self.heads.append(els[0])
        self.tails.append(els[1:])
        Nj = [x for x in els[1:] if x != '<empty>']
        self.M_sigh.append([i for i, x in enumerate(els[1:]) if x == '<empty>'])
        print(Nj)
        self.Nv.append(len(Nj))


  def show(self):
    print("type: {}".format(self.type))
    print("heads: {}".format(self.heads))
    print("tails: {}".format(self.tails))
    print("relations: {}".format(self.relations))
    print("i_dim: {}".format(self.i_dim))
    print("j_dim: {}".format(self.j_dim))
    print("Nv: {}".format(self.Nv))
    print("M_sigh: {}".format(self.M_sigh))

  def get_row_prob(self, pts, qts):
    pT = [1.0] * self.i_dim
    pr_G_Ph_0 = [[(1 / self.i_dim)] * self.i_dim] * self.j_dim
    # print("pr(G=i|Ph=0) = {}".format(pr_G_Ph_0))
    for i in range(self.i_dim):
      for j in range(self.j_dim):
        pr_G_Ph_1 = (1 / self.i_dim) if i in self.M_sigh[j] else ((pts[i][j] / self.Nv[j]) * (1 - (len(self.M_sigh[j]) / self.i_dim)))
        pr_G = (qts[j] * pr_G_Ph_1) + ((1 - qts[j]) * pr_G_Ph_0[j][i])
        print("pr(G) = {}".format(pr_G))
        pT[i] *= pr_G
        print("pT(i) = {}".format(pT[i]))
    return {"pts":pts, "qts":qts, "pT":pT}
    
    
    


if __name__ == '__main__':
  skb = SoftKB()
  skb.show()
  prob = skb.get_row_prob([[0.4,0.2,0.4],[0.3,0.6,0.1]], [0.4,0.2])
  print(prob)
  
