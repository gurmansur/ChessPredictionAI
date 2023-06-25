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
gdd.download_file_from_google_drive(file_id='1koNGmnBxvSzFM0dKDwgxzijjG2pFPKs5',
                                    dest_path='./data_chess.csv'
                                    )
df_Chess = pd.read_csv('data_chess.csv', sep = ",")

# Drop columns
df_Chess = df_Chess[['winner', 'moves', 'white_rating', 'black_rating']]
df_Chess = df_Chess.dropna()

board = chess.Board()

# Transform winner to 0, 1 or 2
df_Chess['winner'] = df_Chess['winner'].map({'white': 0, 'black': 1, 'draw': 2})

# Transform moves to array
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x.split(' '))

#keep 90% of array
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x[0:int(len(x)*0.9)])

#split df 80-20
train = df_Chess[:int(len(df_Chess)*0.8)]
test = df_Chess[int(len(df_Chess)*0.8):]
test = test.reset_index(drop=True)

df_Chessboards_train = pd.DataFrame(columns=['winner', 'board'])
df_Chessboards_test = pd.DataFrame(columns=['winner', 'board'])

for i in range(len(train['moves'])):
  for j in range(len(train['moves'][i])):
    board.push_san(train['moves'][i][j])
    if j % 4 == 0:
      boardString = str(board).replace('\n', ' ')
      df_Chessboards_train = df_Chessboards_train._append({'winner': train['winner'][i], 'board': boardString}, ignore_index=True)
  print(i)
  board.reset()

for i in range(len(test['moves'])):
  for j in range(len(test['moves'][i])):
    board.push_san(test['moves'][i][j])
  boardString = str(board).replace('\n', ' ')
  df_Chessboards_test = df_Chessboards_test._append({'winner': test['winner'][i], 'board': boardString}, ignore_index=True)
  print(i)
  board.reset()


#make column in df_Chess for each element in board
df_Chessboards_train = df_Chessboards_train.join(df_Chessboards_train['board'].str.split(' ', expand=True).add_prefix('board'))
df_Chessboards_test = df_Chessboards_test.join(df_Chessboards_test['board'].str.split(' ', expand=True).add_prefix('board'))

#drop board column
df_Chessboards_train = df_Chessboards_train.drop(columns=['board'])
df_Chessboards_test = df_Chessboards_test.drop(columns=['board'])

#replace letters of pieces with numbers
df_Chessboards_train = df_Chessboards_train.replace('r', 1)
df_Chessboards_train = df_Chessboards_train.replace('n', 2)
df_Chessboards_train = df_Chessboards_train.replace('b', 3)
df_Chessboards_train = df_Chessboards_train.replace('q', 4)
df_Chessboards_train = df_Chessboards_train.replace('k', 5)
df_Chessboards_train = df_Chessboards_train.replace('p', 6)
df_Chessboards_train = df_Chessboards_train.replace('R', 7)
df_Chessboards_train = df_Chessboards_train.replace('N', 8)
df_Chessboards_train = df_Chessboards_train.replace('B', 9)
df_Chessboards_train = df_Chessboards_train.replace('Q', 10)
df_Chessboards_train = df_Chessboards_train.replace('K', 11)
df_Chessboards_train = df_Chessboards_train.replace('P', 12)
df_Chessboards_train = df_Chessboards_train.replace('.', 0)
df_Chessboards_test = df_Chessboards_test.replace('r', 1)
df_Chessboards_test = df_Chessboards_test.replace('n', 2)
df_Chessboards_test = df_Chessboards_test.replace('b', 3)
df_Chessboards_test = df_Chessboards_test.replace('q', 4)
df_Chessboards_test = df_Chessboards_test.replace('k', 5)
df_Chessboards_test = df_Chessboards_test.replace('p', 6)
df_Chessboards_test = df_Chessboards_test.replace('R', 7)
df_Chessboards_test = df_Chessboards_test.replace('N', 8)
df_Chessboards_test = df_Chessboards_test.replace('B', 9)
df_Chessboards_test = df_Chessboards_test.replace('Q', 10)
df_Chessboards_test = df_Chessboards_test.replace('K', 11)
df_Chessboards_test = df_Chessboards_test.replace('P', 12)
df_Chessboards_test = df_Chessboards_test.replace('.', 0)
print(df_Chessboards_train)
print(df_Chessboards_test)

#shuffle rows
df_Chessboards_train = shuffle(df_Chessboards_train)

#split df_Chessboards into x and y
x_train = df_Chessboards_train.drop(columns=['winner'])
y_train = df_Chessboards_train['winner']
y_train = y_train.astype('int')

x_test = df_Chessboards_test.drop(columns=['winner'])
y_test = df_Chessboards_test['winner']
y_test = y_test.astype('int')

#train mlp model
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(1000, 1000, 1000), max_iter=10000)
mlp.fit(x_train, y_train)

#predict
y_pred = mlp.predict(x_test)

#evaluate
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))





