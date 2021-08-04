import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image


def masked_wordcloud(artist_split, mask, min_font = 3, colormap_choice='tab20c'):
    '''
    input:
    returns:
    '''
    text = " ".join(genre for genre in artist_split.genres)
    note_mask = np.array(Image.open(f'../img/{mask}.png'))
    note_mask = note_mask[:,:,2]

    wc = WordCloud(mask=note_mask, min_font_size = min_font, colormap=colormap_choice)
    wc.generate(text)
    wc.to_file(f'../img/{mask}_cloud.png')
    plt.figure(figsize=[15,7])
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()