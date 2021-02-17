from src.data_map import DataMap
from src.featured_question import encode_statement
from src.belief_tracker import BeliefTracker
from src.soft_kb import SoftKB
from src.belief_summary import BeliefSummary
from src.policy_network import PolicyNetwork
from torch.distributions import Categorical
import torch

class KB_infoBot:
  def __init__(self):
    input_file = 'embedding_data/glove.6B.100d.txt'
    data_map = DataMap(input_file)
    self.word_map = data_map.get_map()
    print('word embeddings parsed in the system')
    self.soft_kb = SoftKB()
    # self.p_num = self.soft_kb.j_dim
    self.M = self.soft_kb.j_dim
    self.N = self.soft_kb.i_dim
    self.belief_trackers = [BeliefTracker(100, 256, self.N) for _ in range(self.M)]
    self.belief_summary = BeliefSummary(self.N, self.M, self.soft_kb.M_sigh)
    self.policy_network = PolicyNetwork(2*self.M + 1, 256, self.N)


  def eval(self, statement):
    encoded_question = encode_statement(statement, self.word_map)
    beliefs = [tracker.forward(encoded_question) for tracker in self.belief_trackers]
    pts = [pt[0][0] for (pt, _) in beliefs]
    qts = [qt[0][0] for (_, qt) in beliefs]
    print("pts: {}".format(pts))
    print("qts: {}".format(qts))
    self.soft_kb.show()
    pT = self.soft_kb.get_row_prob(pts, qts)
    print("pT : {}".format(pT))
    h_w = self.belief_summary.forward(pT, pts)
    pT = torch.tensor(pT)
    h_pT = Categorical(probs=pT).entropy()
    print("h(w): {}".format(h_w))
    print("h(pT): {}".format(h_pT))
    s_t = h_w + qts + [h_pT]
    s_t = torch.tensor(s_t)
    print("s_t : {}".format(s_t))
    s = [s_t]
    pi = self.policy_network.forward(s)
    print("pi : {}".format(pi))


