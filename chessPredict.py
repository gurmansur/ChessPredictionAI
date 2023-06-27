import pandas as pd
import re
import numpy as np
import chess
from sklearn.utils import shuffle

pieces = ['p', 'r', 'n', 'b', 'q', 'k']
letter_2_num = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
num_2_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5:'f', 6:'g', 7:'h'}

def create_rep_layer(board, type):
  s = str(board)
  s = re.sub(f'[^{type}{type.upper()} \n]', '.', s)
  s = re.sub(f'{type}', '-1', s)
  s = re.sub(f'[{type.upper()}]', '1', s)
  s = re.sub(f'\.', '0', s)
  
  board_mat = []
  for row in s.split('\n'):
    row = row.split(' ')
    row = [int(i) for i in row]
    board_mat.append(row)
    
  return np.array(board_mat, dtype=object)

def board_2_rep(board):
  layers = []
  for piece in pieces:
    layers.append(create_rep_layer(board, piece))
  board_rep = np.stack(layers, axis=0, dtype=object)
  return board_rep

def move_2_rep(move, board):
  board.push_san(move).uci()
  move = str(board.pop())
  
  from_output_layer = np.zeros((8,8))
  from_row = 8 - int(move[1])
  from_column = letter_2_num[move[0]]
  from_output_layer[from_row, from_column] = 1
  
  to_output_layer = np.zeros((8,8))
  to_row = 8 - int(move[3])
  to_column = letter_2_num[move[2]]
  to_output_layer[to_row, to_column] = 1
  
  return np.stack([from_output_layer, to_output_layer], axis=0)

try:
  df_Chessboards_train = pd.read_csv('df_Chessboards_train.csv', sep = ",")
  df_Chessboards_test = pd.read_csv('df_Chessboards_test.csv', sep = ",")
except:
  # Load data
  df_Chess = pd.read_csv('data_chess.csv', sep = ",")

  # Drop columns
  df_Chess = df_Chess[['winner', 'moves', 'white_rating']]
  df_Chess = df_Chess.dropna()

  #drop´rows with white_rating < 1500
  df_Chess = df_Chess[df_Chess['white_rating'] >= 1500]

  #initialize board
  board = chess.Board()
  print(board_2_rep(board))

  # Transform winner to 0, 1 or 2
  df_Chess['winner'] = df_Chess['winner'].map({'white': 1, 'black': -1, 'draw': 0})

  # Transform moves to array
  df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x.split(' '))

  #drop rows with draws
  df_Chess = df_Chess[df_Chess['winner'] != 0]
  df_Chess = df_Chess.reset_index(drop=True)

  #drop rows with 0 moves
  df_Chess = df_Chess[df_Chess['moves'].map(len) > 0]
  df_Chess = df_Chess.reset_index(drop=True)

  #split df 80-20
  train = df_Chess[:int(len(df_Chess)*0.8)]
  test = df_Chess[int(len(df_Chess)*0.8):]
  test = test.reset_index(drop=True)

  #create dataframes that will recieve the chessboards
  df_Chessboards_train = pd.DataFrame(columns=['winner', 'board'])
  df_Chessboards_test = pd.DataFrame(columns=['winner', 'board'])

  #keep only percentage of moves array
  test['moves'] = test['moves'].apply(lambda x: x[:int(len(x)*0.8)])

  print('Creating training chessboards...')
  #fill train dataframe with string representations of the chessboards
  for i in range(len(train['moves'])):
    for j in range(len(train['moves'][i])):
      board.push_san(train['moves'][i][j])  
      if (j != 0) or j == len(train['moves'][i])-1:  
        boardRep = board_2_rep(board)
        df_Chessboards_train = df_Chessboards_train._append({'winner': train['winner'][i], 'board': boardRep}, ignore_index=True)
    print("\t" + str(round((i/len(train['moves']))*100, 2)) + '% Current Game: ' + str(i), end='\r')
    board.reset()

  print('Creating test chessboards...')
  #fill test dataframe with string representations of the chessboards
  for i in range(len(test['moves'])):
    for j in range(len(test['moves'][i])):
      board.push_san(test['moves'][i][j])
    boardRep = board_2_rep(board)
    df_Chessboards_test = df_Chessboards_test._append({'winner': test['winner'][i], 'board': boardRep}, ignore_index=True)
    print("\t" + str(round((i/len(test['moves']))*100, 2)) + '% Current Game: ' + str(i), end='\r')
    board.reset()

  for i in range(64*6):
    #rewrite the code above with concat
    df_Chessboards_train = pd.concat([df_Chessboards_train, pd.DataFrame(df_Chessboards_train['board'].apply(lambda x: x[int(i/64)][int((i%64)/8)][i%8]).values.tolist(), columns=['piece_' + pieces[int(i/64)] + '_' + str(i%64)])], axis=1)
    df_Chessboards_test = pd.concat([df_Chessboards_test, pd.DataFrame(df_Chessboards_test['board'].apply(lambda x: x[int(i/64)][int((i%64)/8)][i%8]).values.tolist(), columns=['piece_' + pieces[int(i/64)] + '_' + str(i%64)])], axis=1)

  #drop board column
  df_Chessboards_train = df_Chessboards_train.drop(columns=['board'])
  df_Chessboards_test = df_Chessboards_test.drop(columns=['board'])

  #save df_Chessboards to csv
  df_Chessboards_train.to_csv('df_Chessboards_train.csv', index=False)
  df_Chessboards_test.to_csv('df_Chessboards_test.csv', index=False)

print(df_Chessboards_train)
print(df_Chessboards_test)

#drop nan
df_Chessboards_train = df_Chessboards_train.dropna()
df_Chessboards_test = df_Chessboards_test.dropna()

#shuffle df_Chessboards
df_Chessboards_train = shuffle(df_Chessboards_train)
df_Chessboards_test = shuffle(df_Chessboards_test)

#split df_Chessboards into x and y
x_train = df_Chessboards_train.drop(columns=['winner'])
y_train = df_Chessboards_train['winner']
y_train = y_train.astype('int')

x_test = df_Chessboards_test.drop(columns=['winner'])
y_test = df_Chessboards_test['winner']
y_test = y_test.astype('int')

print(x_train)

from joblib import dump, load
#check if mlp.joblib exists
try:
  mlp = load('mlp.joblib')
except:
  #train mlp model
  from sklearn.neural_network import MLPClassifier
  mlp = MLPClassifier(hidden_layer_sizes=(10), max_iter=100, alpha=0.0001, solver='adam', verbose=True, n_iter_no_change=2, tol=0.001)
  mlp.fit(x_train, y_train)
  dump(mlp, 'mlp.joblib')

#predict
predictions = mlp.predict(x_test)

#evaluate
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))