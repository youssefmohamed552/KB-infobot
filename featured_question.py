def encode_statement(statement, word_map):
  encoded_statement = []
  count = 0
  segmented_question =  statement.strip().split(' ')
  for word in segmented_question: 
    try:
      encoded_statement.append(word_map[word])
    except KeyError:
      count += 1 

  print('encoded statement with accuracy {}'.format((1.0 - (count / len(segmented_question))) * 100.0))

  return encoded_statement
    
