import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from google_drive_downloader import GoogleDriveDownloader as gdd
import chess

# Load data
df_Chess = pd.read_csv('data_chess.csv', sep = ",")

# Drop columns
df_Chess = df_Chess[['winner', 'moves']]
df_Chess = df_Chess.dropna()

#initialize board
board = chess.Board()

# Transform winner to 0, 1 or 2
df_Chess['winner'] = df_Chess['winner'].map({'white': 0, 'black': 1, 'draw': 2})

# Transform moves to array
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x.split(' '))

#keep only 90% of array
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x[:int(len(x)*0.8)])

#drop rows with draws
df_Chess = df_Chess[df_Chess['winner'] != 2]
df_Chess = df_Chess.reset_index(drop=True)

#drop rows with 0 moves
df_Chess = df_Chess[df_Chess['moves'].map(len) > 0]
df_Chess = df_Chess.reset_index(drop=True)

#split df 80-20
train = df_Chess[:int(len(df_Chess)*0.8)]
test = df_Chess[int(len(df_Chess)*0.8):]
test = test.reset_index(drop=True)

#check if df_Chessboards exists
try:
  df_Chessboards_train = pd.read_csv('df_Chessboards_train.csv', sep = ",")
  df_Chessboards_test = pd.read_csv('df_Chessboards_test.csv', sep = ",")
except:
  #create dataframes that will recieve the chessboards
  df_Chessboards_train = pd.DataFrame(columns=['winner', 'board'])
  df_Chessboards_test = pd.DataFrame(columns=['winner', 'board'])

  #fill train dataframe with string representations of the chessboards
  for i in range(len(train['moves'])):
    for j in range(len(train['moves'][i])):
      board.push_san(train['moves'][i][j])
      if (j % 7 == 0 and j != 0) or j == len(train['moves'][i])-1:
        boardString = str(board).replace('\n', ' ')
        df_Chessboards_train = df_Chessboards_train._append({'winner': train['winner'][i], 'board': boardString}, ignore_index=True)
    board.reset()

  #fill test dataframe with string representations of the chessboards
  for i in range(len(test['moves'])):
    for j in range(len(test['moves'][i])):
      board.push_san(test['moves'][i][j])
    boardString = str(board).replace('\n', ' ')
    df_Chessboards_test = df_Chessboards_test._append({'winner': test['winner'][i], 'board': boardString}, ignore_index=True)
    board.reset()

  #make column in df_Chessboards for each element in board
  df_Chessboards_train = df_Chessboards_train.join(df_Chessboards_train['board'].str.split(' ', expand=True).add_prefix('board'))
  df_Chessboards_test = df_Chessboards_test.join(df_Chessboards_test['board'].str.split(' ', expand=True).add_prefix('board'))

  #drop board column
  df_Chessboards_train = df_Chessboards_train.drop(columns=['board'])
  df_Chessboards_test = df_Chessboards_test.drop(columns=['board'])

  #save df_Chessboards to csv
  df_Chessboards_train.to_csv('df_Chessboards_train.csv', index=False)
  df_Chessboards_test.to_csv('df_Chessboards_test.csv', index=False)

#load df_Chessboards from csv
df_Chessboards_train = pd.read_csv('df_Chessboards_train.csv', sep = ",")
df_Chessboards_test = pd.read_csv('df_Chessboards_test.csv', sep = ",")

#one hot encode the board columns
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(df_Chessboards_train.drop(columns=['winner']))
df_Chessboards_train_encoded = enc.transform(df_Chessboards_train.drop(columns=['winner'])).toarray()
df_Chessboards_test_encoded = enc.transform(df_Chessboards_test.drop(columns=['winner'])).toarray()

#add winner column to encoded dataframes
df_Chessboards_train_encoded = pd.DataFrame(df_Chessboards_train_encoded)
df_Chessboards_train_encoded['winner'] = df_Chessboards_train['winner']
df_Chessboards_test_encoded = pd.DataFrame(df_Chessboards_test_encoded)
df_Chessboards_test_encoded['winner'] = df_Chessboards_test['winner']

#df_chessboards = encoded dataframes
df_Chessboards_train = df_Chessboards_train_encoded
df_Chessboards_test = df_Chessboards_test_encoded
print(df_Chessboards_train)
print(df_Chessboards_test)

#shuffle df_Chessboards
df_Chessboards_train = shuffle(df_Chessboards_train)

#split df_Chessboards into x and y
x_train = df_Chessboards_train.drop(columns=['winner'])
y_train = df_Chessboards_train['winner']

x_test = df_Chessboards_test.drop(columns=['winner'])
y_test = df_Chessboards_test['winner']

from joblib import dump, load
#check if mlp.joblib exists
try:
  mlp = load('mlp.joblib')
except:
  #train mlp model
  from sklearn.neural_network import MLPClassifier
  mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=100, alpha=0.0001, solver='adam', random_state=17, verbose=True, n_iter_no_change=2, tol=0.001)
  mlp.fit(x_train, y_train)
  dump(mlp, 'mlp.joblib')

#predict
predictions = mlp.predict(x_test)

#evaluate
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))