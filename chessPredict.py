import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import chess
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from google_drive_downloader import GoogleDriveDownloader as gdd

# Load data
gdd.download_file_from_google_drive(file_id='1koNGmnBxvSzFM0dKDwgxzijjG2pFPKs5',
                                    dest_path='./data_chess.csv'
                                    )
df_Chess = pd.read_csv('data_chess.csv', sep = ",")

# Drop columns
df_Chess = df_Chess[['winner', 'moves']]
df_Chess = df_Chess.dropna()

# Use 5% of dataset only
df_Chess = df_Chess.sample(frac=0.05)

# Transform winner to 0, 1 or 2
df_Chess['winner'] = df_Chess['winner'].map({'white': 0, 'black': 1, 'draw': 2})

# Create board
board = chess.Board()

# Transform moves to array
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x.split(' '))

# Get 40% of moves
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x[:int(len(x)*0.4)])

# Create board for each move
def create_board(moves):
    for i in range(len(moves)):
        try:
            board.push_san(moves[i])
        except:
            pass
    return board.copy()

df_Chess['moves'] = df_Chess['moves'].apply(lambda x: create_board(x))

# Create new columns for each place on the board
for i in range(64):
    df_Chess['piece_' + str(i)] = df_Chess['moves'].apply(lambda x: x.piece_at(i).piece_type if x.piece_at(i) != None else 0)

# Split train and test using 20% of data for testing
from sklearn.model_selection import train_test_split

X = df_Chess.drop(['winner', 'moves'], axis=1)
y = df_Chess['winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train classifier model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))