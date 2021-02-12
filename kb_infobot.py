from data_map import DataMap
from featured_question import encode_statement
from belief_tracker import BeliefTracker

class KB_infoBot:
  def __init__(self):
    input_file = 'glove.6B.100d.txt'
    data_map = DataMap(input_file)
    self.word_map = data_map.get_map()
    print('word embeddings parsed in the system')
    self.belief_tracker = BeliefTracker(100, 256)

  def eval(self, statement):
    encoded_question = encode_statement(statement, self.word_map)
    belief = self.belief_tracker.forward(encoded_question)
    print(belief)
    
