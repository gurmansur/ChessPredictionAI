import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
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

# Transform winner to 0, 1 or 2
df_Chess['winner'] = df_Chess['winner'].map({'white': 0, 'black': 1, 'draw': 2})

# Transform moves to numbers
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x.replace('a', '1'))
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x.replace('b', '2'))
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x.replace('c', '3'))
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x.replace('d', '4'))
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x.replace('e', '5'))
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x.replace('f', '6'))
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x.replace('g', '7'))
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x.replace('h', '8'))
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x.replace('x', '9'))
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x.replace('=', '10'))
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x.replace('+', '11'))
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x.replace('#', '12'))
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x.replace('O-O-O', '14'))
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x.replace('O-O', '13'))
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x.replace('Q', '15'))
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x.replace('R', '16'))
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x.replace('N', '17'))
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x.replace('B', '18'))
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x.replace('K', '19'))
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x.replace('P', '20'))

# Transform moves to array
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x.split(' '))

# Get 40% of moves
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x[:int(len(x)*0.4)])

# Set lenght of all moves array to same value
df_Chess['moves'] = df_Chess['moves'].apply(lambda x: x + ['0']*(max([len(i) for i in df_Chess['moves']]) - len(x)))

# Split train and test using 20% of data for testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_Chess['moves'], df_Chess['winner'], test_size=0.2)

X_train = np.array(X_train.tolist())
X_test = np.array(X_test.tolist())

# Train SVC model
import sklearn.svm as svm

model = svm.SVC(kernel='linear', C=1, gamma=1)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))