class DataMap:
  def __init__(self, file_name):
    self.file_name = file_name

  def get_map(self):
    word_map = {}
    with open(self.file_name, 'r') as fp:
      for line in fp:
        tokens = line.strip().split(' ')
        word_map[tokens[0]] = [float(token) for token in tokens[1:]]
    return word_map
    
