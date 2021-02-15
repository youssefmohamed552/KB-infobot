from src.data_map import DataMap
from src.featured_question import encode_statement
from src.belief_tracker import BeliefTracker
from src.soft_kb import SoftKB

class KB_infoBot:
  def __init__(self):
    input_file = 'embedding_data/glove.6B.100d.txt'
    data_map = DataMap(input_file)
    self.word_map = data_map.get_map()
    print('word embeddings parsed in the system')
    self.soft_kb = SoftKB()
    # self.p_num = self.soft_kb.j_dim
    self.belief_trackers = [BeliefTracker(100, 256, self.soft_kb.j_dim) for _ in range(self.soft_kb.i_dim)]


  def eval(self, statement):
    encoded_question = encode_statement(statement, self.word_map)
    beliefs = [tracker.forward(encoded_question) for tracker in self.belief_trackers]
    pts = [pt[0][0] for (pt, _) in beliefs]
    qts = [qt[0][0] for (_, qt) in beliefs]
    print("pts: {}".format(pts))
    print("qts: {}".format(qts))
    self.soft_kb.show()
    st = self.soft_kb.get_row_prob(pts, qts)

    print("state : {}".format(st))

