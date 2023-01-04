"""
CSC110 Fall 2020 Project 1, Part 1 Open Json File

Description
===============================
This module is dedicated to visualizing our computed code through word clouds and a pie chart.


Copyright and Usage Information
===============================
This file is provided solely for the personal and private use of the professors and TAs
in CSC110 at the University of Toronto St. George campus. All forms of
distribution of this code, whether as given or with any changes, are
expressly prohibited. For more information on copyright for this CSC110 project,
please consult with us.

This file is Copyright (c) 2020 Jiajin Wu, Ehsan Emadi, Ashkan Aleshams and Michael Galloro.
"""

from typing import Dict, List

# graphing libraries
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
import numpy as np
from PIL import Image

import computing_data


def locations_wordcloud(words: Dict[str, int], image: str) -> None:
    """
    Plots a word cloud designed for words related to twitter user locations.
    """
    # Generate a word cloud image
    earth = np.array(Image.open(image))
    # set a min font size and a max font size, and how the size of a word in word cloud scales
    # so the graph won't look too ridiculous
    earth_word_cloud = WordCloud(background_color="white", mode="RGBA", mask=earth,
                                 min_font_size=0, max_font_size=50,
                                 relative_scaling=0.5).generate_from_frequencies(words)

    # create coloring from image
    image_colors = ImageColorGenerator(earth)

    plt.figure(figsize=[11, 11])
    # Let word cloud have the colour of the picture
    plt.imshow(earth_word_cloud.recolor(color_func=image_colors), interpolation="bilinear")
    # creates a title
    plt.suptitle('About The Tweeters: Where Do They Live?', size='xx-large',
                 y=0.92, weight='bold', family='sans-serif')
    plt.axis("off")

    plt.show()


def tweet_words_wordcloud(words: Dict[str, int], image: str) -> None:
    """
    Plots a word cloud designed for words related to words in tweets.
    """
    # Generate a word cloud image
    speech = np.array(Image.open(image))
    # set a min font size and a max font size, and how the size of singular word scales
    # so the graph won't look too ridiculous
    speech_word_cloud = WordCloud(background_color="white", mode="RGBA", mask=speech,
                                  min_font_size=2, max_font_size=50,
                                  relative_scaling=0.5).generate_from_frequencies(words)

    # create coloring from image
    image_colors = ImageColorGenerator(speech)

    plt.figure(figsize=[10, 10])
    # Let word cloud have the colour of the picture
    plt.imshow(speech_word_cloud.recolor(color_func=image_colors), interpolation="bilinear")
    # creates a title
    plt.suptitle('About The Tweets: What Are They Saying?', size='xx-large',
                 y=0.87, weight='bold', family='sans-serif')
    plt.axis("off")

    plt.show()


def descriptions_wordcloud(words: Dict[str, int], image: str) -> None:
    """
    Plots a word cloud designed for words related to twitter user descriptions.
    """
    # Generate a word cloud image
    brain = np.array(Image.open(image))
    # set a min font size and a max font size, and how the size of singular word scales
    brain_word_cloud = WordCloud(background_color="white", mode="RGBA", mask=brain,
                                 min_font_size=2, max_font_size=50,
                                 relative_scaling=0.5).generate_from_frequencies(words)

    # create coloring from image
    image_colors = ImageColorGenerator(brain)

    plt.figure(figsize=[10, 10])
    # Let word cloud have the colour of the picture
    plt.imshow(brain_word_cloud.recolor(color_func=image_colors), interpolation="bilinear")
    # creates a title
    plt.suptitle('About The Tweeters: How Do They Describe Themselves?', size='xx-large', y=0.89,
                 weight='bold', family='sans-serif')
    plt.axis("off")

    plt.show()


def hashtags_wordcloud(words: Dict[str, int], image: str) -> None:
    """
    Draws a word cloud designed for words related to tweet hashtags.
    """
    trim_words = computing_data.trim_max_values(words, 10)
    # Generate a word cloud image
    bird = np.array(Image.open(image))
    # set a min font size and a max font size, and how the size of singular word scales
    bird_word_cloud = WordCloud(background_color="white", mode="RGBA", mask=bird,
                                min_font_size=2, max_font_size=50,
                                relative_scaling=0.5).generate_from_frequencies(trim_words)

    # create coloring from image
    image_colors = ImageColorGenerator(bird)

    plt.figure(figsize=[10, 10])
    # Let word cloud have the colour of the picture
    plt.imshow(bird_word_cloud.recolor(color_func=image_colors), interpolation="bilinear")
    # creates a title
    plt.suptitle('About The Tweets: What Are The Hashtags?', size='xx-large', y=0.89,
                 weight='bold', family='sans-serif')
    plt.axis("off")

    plt.show()


def vader_piechart(vader_info: List) -> None:
    """
    Draws a piechart designed for VADER sentiment analysis data.
    """

    labels = 'Extremely Positive', 'Positive', 'Mildly Positive', 'Mildly Negative', 'Negative', \
             'Extremely Negative', 'Neutral'
    # uses get_vader_score_ratios to compute sizes of each respective pie
    sizes = computing_data.get_vader_score_ratios(vader_info)
    explode = (0, 0.1, 0, 0, 0, 0, 0)  # "explode" designates which pie slightly slides
    # out of frame

    _, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    plt.suptitle('About The Tweets: How Positive Are They? (VADER)',
                 size='medium', y=0.95, weight='bold', family='sans-serif')

    plt.show()


def visualization_context(vader_info: List) -> None:
    """
    Displays the tweets witht he maximum and minimum vader scores.

    Preconditions:
     - len(vader_info[1].tweet) < 350
     - len(vader_info[2].tweet) < 350

    """
    plt.figure(figsize=[11, 4])
    # draws words on the figure
    plt.figtext(x=0.02, y=0.9, s='Read First: Intructions For Best Use', fontsize='x-large')
    plt.figtext(x=0.02, y=0.85, s='Please note that there are four word clouds and one pie chart. '
                                  'Feel free to use the zoom function to explore words within '
                                  'each cloud.')
    plt.figtext(x=0.02, y=0.79, s='You may need to close each figure window (once you are finished'
                                  ' viewing it) in order to view the next one. Including this one!')

    plt.figtext(x=0.02, y=0.72, s='About Vader', fontsize='x-large')
    plt.figtext(x=0.02, y=0.66, s='VADER (Valence Aware Dictionary and sEntiment Reasoner) '
                                  'uses a complex algorithm to assign "positivity" ratings to '
                                  'sentences.')
    plt.figtext(x=0.02, y=0.61, s='The library has a very long run time for larger data sets, so'
                                  ' as a compromise, we computed on a sub-set of our tweets.')
    plt.figtext(x=0.02, y=0.55, s='To help show how VADER works, here are the two tweets '
                                  'with the highest and lowest rating in our sample size.')
    plt.figtext(x=0.02, y=0.5, s='We take no responsibility for what they may say!')

    plt.figtext(x=0.02, y=0.38, s='Tweet with highest positivity rating', fontsize='large')

    # for displaying highest positivity tweet:
    if len(vader_info[1].tweet) < 135:
        plt.figtext(x=0.02, y=0.32, s=vader_info[1].tweet, fontsize='small', color='green')
    else:
        plt.figtext(x=0.02, y=0.32, s=''.join([vader_info[1].tweet[letter] for letter in
                                               range(0, 135)]) + '-', fontsize='small',
                    color='green')
        plt.figtext(x=0.02, y=0.28, s=''.join([vader_info[1].tweet[letter] for letter in
                                               range(135, len(vader_info[1].tweet))]),
                    fontsize='small', color='green')
    # for displaying lowest positivity tweet:
    plt.figtext(x=0.02, y=0.18, s='Tweet with lowest positivity rating', fontsize='large')
    if len(vader_info[2].tweet) < 135:
        plt.figtext(x=0.02, y=0.07, s=vader_info[2].tweet, fontsize='small', color='red')
    else:
        plt.figtext(x=0.02, y=0.12, s=''.join([vader_info[2].tweet[letter] for letter in
                                               range(0, 135)]) + '-', fontsize='small',
                    color='red')
        plt.figtext(x=0.02, y=0.07, s=''.join([vader_info[2].tweet[letter] for letter in
                                               range(135, len(vader_info[2].tweet))]),
                    fontsize='small', color='red')

    plt.show()


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['python_ta.contracts', 'typing', 'computing_data', 'matplotlib.pyplot', 'wordcloud',
                          'numpy', 'PIL'],
        'max-line-length': 150,
        'disable': ['R1705', 'C0200']
    })

    import doctest

    doctest.testmod()
