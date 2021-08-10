import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

def masked_wordcloud(artist_split, mask, name, min_font = 3, colormap_choice='tab20c'):
    '''
    input:
    returns:
    '''
    text = " ".join(genre for genre in artist_split.genres)
    note_mask = np.array(Image.open(mask))
    note_mask = note_mask[:,:,2]

    wc = WordCloud(mask=note_mask, min_font_size = min_font, colormap=colormap_choice)
    wc.generate(text)
    
    img_path = '/content/drive/MyDrive/Colab Notebooks/spotify_genre/img/'
    plt.figure(figsize=[15,7])
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(f'{img_path}{name}')