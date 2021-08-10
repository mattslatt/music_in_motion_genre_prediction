import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

models_dir = '/content/drive/MyDrive/Colab Notebooks/spotify_genre/models'

def clean_artists(data):
    '''
    input: raw artist dataframe
    returns: cleaned artist dataframe with genre counts
    * remove artists with no genre tags
    * clean genre tags of brackets and quotes
    * count number of tagged genres per artist
    * split artists with multiple genre tags
    * remove non-musical genres (reading, hoorspiel, etc.)
    * 
    '''
    non_music = ['reading','hoerspiel','kleine hoerspiel','kindermusik','barnsagor']
    artist_genres = data[data['genres'] != '[]'].copy()
    artist_genres['genres_clean'] = artist_genres['genres'].str.strip("[]")
    artist_genres['genre_count'] = artist_genres['genres'].str.count(',') + 1
    artist_split = artist_genres.assign(genres=artist_genres['genres_clean'].str.split(', ')).explode('genres')
    artist_split['genres'] = artist_split['genres'].str.strip("'")
    artist_split['id_artists'] = artist_split['id']
    artist_split.drop(columns='genres_clean')
    artist_split = artist_split[~artist_split['genres'].isin(non_music)]
    print('{} total artists'.format(data.shape[0]))
    print('{} artists with labeled genres'.format(artist_genres.shape[0]))
    print('{} genre-artist pairs after processing'.format(artist_split.shape[0]))
    return artist_split

def clean_tracks(data, max_artists=3):
    '''
    input: raw tracks dataframe
    returns: cleaned tracks dataframe with tracks and their attributed artists,
    duplicating tracks with multiple tagged artists
    * remove tracks with no artist id
    * clean artist ids of brackets
    * artist count represents number of artists attributed to a given track
    * remove tracks with more than 3 affiliated artists
    * split tracks for each tagged artist
    '''
    tracks = data[data['id_artists'] != '[]'].copy()
    tracks['id_artists_clean'] = tracks['id_artists'].str.strip("[]")
    tracks['artist_count'] = tracks['id_artists_clean'].str.count(',') + 1
    tracks_max_artists = tracks[tracks['artist_count'] <= max_artists]
    tracks_split = tracks_max_artists.assign(id_artists=tracks_max_artists['id_artists_clean'].str.split(', ')).explode('id_artists')
    tracks_split['id_artists'] = tracks_split['id_artists'].str.strip("'")
    tracks_split.drop(columns=['id_artists_clean'], inplace=True)
    print('{} total tracks'.format(data.shape[0]))
    print('{} tracks with {} or fewer artists'.format(tracks_max_artists.shape[0], max_artists))
    print('{} track-artist pairs after processing'.format(tracks_split.shape[0]))
    return tracks_split

def top_x_genres(artist_split, tracks_split, num_genres, obscurity=0):
  '''
  input: cleaned artist dataframe, tracks dataframe, number of most popular genres 
  to include, and obscurity (number of most popular genres to remove from the top of the list)
  returns: list of most popular genres, artist data containing only those genres
  '''
  tracks_artists = artist_split.merge(tracks_split, on='id_artists')
  top_x = tracks_artists['genres'].value_counts()[obscurity:num_genres+obscurity].index
  artist_x_genres = tracks_artists[tracks_artists['genres'].isin(top_x)]
  if obscurity == 0:
    print(f'{artist_x_genres.shape[0]} tracks from top {num_genres} genres')
  else:
    print(f'{artist_x_genres.shape[0]} tracks from top {num_genres} obscure genres')
  return artist_x_genres





