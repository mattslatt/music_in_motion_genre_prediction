import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def clean_artists(data):
    artist_genres = data[data['genres'] != '[]'].copy()
    artist_genres['genres_clean'] = artist_genres['genres'].str.strip("[]")
    artist_genres['genre_count'] = artist_genres['genres'].str.count(',') + 1
    artist_split = artist_genres.assign(genres=artist_genres['genres_clean'].str.split(', ')).explode('genres')
    artist_split['genres'] = artist_split['genres'].str.strip("'")
    artist_split['id_artists'] = artist_split['id']
    artist_split.drop(columns='genres_clean')
    print('{} artists total'.format(data.shape[0]))
    print('{} artists with labeled genres'.format(artist_genres.shape[0]))
    print('{} genre-artist pairs after processing'.format(artist_split.shape[0]))
    return artist_split

def clean_tracks(data, max_artists=3):
    tracks = data[data['id_artists'] != '[]'].copy()
    tracks['id_artists_clean'] = tracks['id_artists'].str.strip("[]")
    tracks['artist_count'] = tracks['id_artists_clean'].str.count(',') + 1
    tracks_max_artists = tracks[tracks['artist_count'] <= max_artists]
    tracks_split = tracks_max_artists.assign(id_artists=tracks_max_artists['id_artists_clean'].str.split(', ')).explode('id_artists')
    tracks_split['id_artists'] = tracks_split['id_artists'].str.strip("'")
    tracks_split.drop(columns=['id_artists_clean'], inplace=True)
    print('{} tracks total'.format(data.shape[0]))
    print('{} tracks with {} or fewer artists'.format(tracks_max_artists.shape[0], max_artists))
    print('{} track-artist pairs after processing'.format(tracks_split.shape[0]))
    return tracks_split

def top_x_genres(artist_split, num_genres = 50):
    '''
    input: processsed artist data (dataframe), number of most popular genres 
    to include (int, default 50)
    returns: list of most popular genres, artist data containing only those genres
    '''
    top_x = artist_split['genres'].value_counts()[:num_genres].index
    artist_x_genres = artist_split[artist_split['genres'].isin(top_x)]
    return top_x, artist_x_genres

def track_features_and_labels(data):
    tracks_artists_clean = data[['genres','danceability',
       'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
       'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']]
    X, y = tracks_artists_clean.iloc[:, 1:], tracks_artists_clean.iloc[:, :1]
    return X, y