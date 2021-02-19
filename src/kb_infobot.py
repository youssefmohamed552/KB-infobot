from src.arch.data_map import DataMap
from src.arch.featured_question import encode_statement
from src.arch.belief_tracker import BeliefTracker
from src.arch.soft_kb import SoftKB
from src.arch.belief_summary import BeliefSummary
from src.arch.policy_network import PolicyNetwork
from src.arch.action_selection import ActionSelection
from torch.distributions import Categorical
import torch

class KB_infoBot:
  def __init__(self):
    input_file = 'embedding_data/glove.6B.100d.txt'
    data_map = DataMap(input_file)
    self.R = 5
    self.hidden_dim = 256
    self.input_dim = 100
    self.H = 1
    self.t = 0
    self.s = []
    self.r = []
    self.word_map = data_map.get_map()
    print('word embeddings parsed in the system')
    self.soft_kb = SoftKB()
    self.M = self.soft_kb.j_dim
    self.N = self.soft_kb.i_dim
    self.belief_trackers = [BeliefTracker(self.input_dim, self.hidden_dim, self.N) for _ in range(self.M)]
    self.belief_summary = BeliefSummary(self.N, self.M, self.soft_kb.M_sigh)
    self.policy_network = PolicyNetwork(2*self.M + 1, self.hidden_dim, self.N)
    self.action_selection = ActionSelection()


  def run(self):
    for _ in range(self.H):
      question = 'where are the pyramids'
      self.eval(question)
    print("state of the system : {}".format(self.s))
    print("system rewards : {}".format(self.r))


  def eval(self, statement):
    encoded_question = encode_statement(statement, self.word_map)
    beliefs = [tracker.forward(encoded_question) for tracker in self.belief_trackers]
    pts = [pt[0][0] for (pt, _) in beliefs]
    qts = [qt[0][0] for (_, qt) in beliefs]
    pT = self.soft_kb.get_row_prob(pts, qts)
    h_w = self.belief_summary.forward(pT, pts)
    pT = torch.tensor(pT)
    h_pT = Categorical(probs=pT).entropy()
    st = torch.tensor(h_w + qts + [h_pT])
    self.s.append(st)
    pi = self.policy_network.forward(self.s)
    self.action_selection.form_I(2, pT)
    self.action_selection.reward(pT)
    self.r.append(self.action_selection.r)
    mean_I = self.action_selection.get_mean(pT)
    print("mean(I) : {}".format(mean_I))
    self.t += 1


