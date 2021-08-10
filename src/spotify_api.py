import requests
import pandas as pd
from pprint import pprint

def request_track_features(access_token):
  response = requests.get(
    SPOTIFY_TRACK_URL,
    headers = {
      'Authorization': f'Bearer {access_token}'
    }
  )
  resp_json = response.json()
  feature_df = pd.DataFrame(list(resp_json.items()), columns = ['feature','value'])
  feature_df.set_index('feature', inplace=True)
  feature_df = pd.DataFrame.transpose(feature_df)
  return feature_df

def url_to_features():
    track_url = input('Copied song link \n(right click on song -> share): ')
    track_id = track_url.split('/')[4].split('?')[0]

    global SPOTIFY_TRACK_URL
    SPOTIFY_TRACK_URL = f'https://api.spotify.com/v1/audio-features/{track_id}'
    SPOTIFY_ACCESS_TOKEN = input('Spotify Access Token:')
    
    current_track_info = request_track_features(SPOTIFY_ACCESS_TOKEN)
    pprint(current_track_info)
    return current_track_info

if __name__ == '__main__':
  url_to_features()