import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

models_dir = '/content/drive/MyDrive/Colab Notebooks/spotify_genre/models'

def track_features_and_labels(data):
    tracks_artists_clean = data[['genres','danceability',
       'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
       'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']]
    X, y = tracks_artists_clean.iloc[:, 1:], tracks_artists_clean.iloc[:, :1]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test

def model_train_save(model, tracks_artists, save_name, class_report = False):
    X_train, X_test, y_train, y_test = track_features_and_labels(tracks_artists)
    model.fit(X_train, y_train.values.ravel())
    if class_report:
      pred = model.predict(X_test)
      print(classification_report(y_test, pred))
    with open(f'{models_dir}/{save_name}.pkl', 'wb') as m:
      pickle.dump(model, m)
    print(f'Model saved as {save_name}.pkl')
    print(f'{save_name} accuracy: {round(model.score(X_test, y_test),2)}')


def model_testing(model, tracks_artists):
    X_train, X_test, y_train, y_test = track_features_and_labels(tracks_artists)
    model.fit(X_train, y_train.values.ravel())
    pred = model.predict(X_test)
    print(classification_report(y_test, pred))

def track_genre_predictions(track, model, num_genres = 3):
    '''
    input: single track from tracks_artists dataframe, using iloc[]
    returns: list of most likely genres
    '''
    track_features = track[['danceability',
       'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
       'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']]
    classes = model.classes_
    scores = model.predict_proba(np.array(track_features).reshape(1,-1))[0]
    genre_predictions = classes[np.argsort(scores)[::-1]][:num_genres]
    print(f'Genres to explore: \n{genre_predictions}')
    return genre_predictions