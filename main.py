import sys
from data_map import DataMap
from featured_question import encode_statement
from belief_tracker import BeliefTracker

def main():
  input_file = 'glove.6B.100d.txt'
  data_map = DataMap(input_file)
  word_map = data_map.get_map()
  print('word embeddings parsed in the system')
  question = 'where are the pyramids'
  encoded_question = encode_statement(question, word_map)
  belief_tracker = BeliefTracker(100, 256)
  belief = belief_tracker.forward(encoded_question)
  print(belief)


  


    

if __name__ == '__main__':
  main()
