import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='1koNGmnBxvSzFM0dKDwgxzijjG2pFPKs5',
                                    dest_path='./data_chess.csv'
                                    )
df_Chess = pd.read_csv('data_chess.csv', sep = ",")
df_Chess = df_Chess[['opening_eco', 'turns', 'winner', 'moves']]
df_Chess = df_Chess.dropna()
