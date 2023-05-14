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

# Transform moves to one-hot encoding
df_Chess = pd.get_dummies(df_Chess, columns=['moves'])

# Transform winner to 0 or 1
df_Chess['winner'] = df_Chess['winner'].apply(lambda x: 1 if x == 'white' else 0)

# Shuffle data
df_Chess = shuffle(df_Chess)

# Split data
X = df_Chess.drop(['winner'], axis=1)
y = df_Chess['winner']

# Split data into train and test
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

# Scale data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SGDClassifier
from sklearn.linear_model import SGDClassifier
model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
print(model.score(X_test, y_test))