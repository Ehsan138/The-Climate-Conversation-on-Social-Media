"""
CSC110 Fall 2020 Project 1, Part 1 Open Json File

Description
===============================
This module is dedicated to computing the raw dictionary data taken from the JSON
tweet file, by aggregating, transforming, and filtering it into code that is
ready for visualization!

Copyright and Usage Information
===============================
This file is provided solely for the personal and private use of the professors and TAs
in CSC110 at the University of Toronto St. George campus. All forms of
distribution of this code, whether as given or with any changes, are
expressly prohibited. For more information on copyright for this CSC110 project,
please consult with us.

This file is Copyright (c) 2020 Jiajin Wu, Ehsan Emadi, Ashkan Aleshams and Michael Galloro.
"""

from typing import List, Dict

import vaderSentiment
# necessary for some implementations of vader sentiment library:
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import reading_data
import stopwords_and_redundancies


def accumulate_tweet_words(tweets: List[reading_data.Tweet]) -> Dict[str, int]:
    """
    Returns a dict with words in tweets as keys and how often they appear as values.
    """
    # tweet word accumulator
    tweet_words_so_far = {}

    for twit in tweets:
        tweet = clean_words(twit.tweet)  # cleans the words that we don't want

        for word in tweet:
            # adds a key:value pair to tweet_words_so_far if the key does not exist
            if word not in tweet_words_so_far:
                tweet_words_so_far[word] = 1
            # increases tweet word value by one if the key already exist
            else:
                tweet_words_so_far[word] += 1

    return tweet_words_so_far


def accumulate_locations(tweets: List[reading_data.Tweet]) -> Dict[str, int]:
    """
    Returns a dict with twitter user locations as keys and how often they appear as values.
    """
    # location accumulator
    locations_so_far = {}

    for tweet in tweets:
        # removes the words that we don't want from user location
        # description
        locations = clean_locations(clean_words(tweet.user.location))

        for location in locations:
            # adds a key:value pair to locations_so_far if the key does not exist
            if location not in locations_so_far:
                locations_so_far[location] = 1
            # increases location value by one if the key already exist
            else:
                locations_so_far[location] += 1

    return locations_so_far


def accumulate_descriptions(tweets: List[reading_data.Tweet]) -> Dict[str, int]:
    """
    Returns a dict with twitter user descriptions as keys and how often they appear as values.
    """
    # description accumulator
    descriptions_so_far = {}

    for tweet in tweets:
        # removes words in user descriptions that we don't want
        description = clean_words(tweet.user.description)

        for word in description:
            if len(word) >= 4:  # so we don't have meaningless descriptions
                # adds a key:value pair to descriptions_so_far if the key
                # does not exist
                if word not in descriptions_so_far:
                    descriptions_so_far[word] = 1
                # increases description word value by one if the key already exists
                else:
                    descriptions_so_far[word] += 1

    return descriptions_so_far


def accumulate_hashtags(tweets: List[reading_data.Tweet]) -> Dict[str, int]:
    """
    Returns a dict with twitter hashtags as keys and how often they appear as values.
    """
    # hashtag accumulator
    hashtags_so_far = {}

    for tweet in tweets:
        hashtags = tweet.hashtags

        for word in hashtags:
            # adds a key:value pair to hashtags_so_far if the key does not exist
            hashtag = word.lower()
            if hashtag not in hashtags_so_far:
                hashtags_so_far[hashtag] = 1
            # increase hashtag value by one if the key already exists
            else:
                hashtags_so_far[hashtag] += 1

    return hashtags_so_far


def clean_locations(locations: List[str]) -> List[str]:
    """
    Return a list of unified location terms, and removes many common user location-related
    redundancies, such as calling the same place by a different name, or common non-locations such as
    "city".

    >>> clean_locations(['city', 'as', 'usa', 'ny', 'new', 'york', 'the', 'states'])
    ['usa', 'new york', 'new york', 'usa']
    """
    for i in range(0, len(locations)):
        # so meaningless words won't show up as locations
        if len(locations[i]) < 4 and locations[i] != 'uk' and locations[i] != 'usa' and \
                locations[i] != 'ny':
            locations[i] = ''
        # so meaningless locations won't show up
        elif locations[i] in stopwords_and_redundancies.LOCATION_REDUNDANCIES:
            locations[i] = ''
        # unify the names of the USA
        elif locations[i] == 'states' or locations[i] == 'america':
            locations[i] = 'usa'
        # unify the names of the UK
        elif locations[i] == 'kingdom':
            locations[i] = 'uk'
        # unify the names of New York
        elif locations[i] == 'york' or locations[i] == 'ny':
            locations[i] = 'new york'
    # remove the empty strings
    while '' in locations:
        locations.remove('')

    return locations


def clean_words(words: str) -> List[str]:
    """
    Return a list of strings that is a list of each word in the input words.
    The function also removes words that are considered stopwords.
    Each of these returned words are all lower case and only
    contain alphanumeric characters.

    >>> clean_words('a global war$ming bad for business')
    ['bad', 'business']
    """
    string_so_far = ''
    string_lower = words.lower()  # make the words lower cased

    for character in string_lower:
        # ignores characters that are not spaces or alphanumeric.
        if character.isalpha() or character == ' ':
            string_so_far = string_so_far + character
    # split the string into lists
    list_of_words = string_so_far.split(' ')

    non_common_words_so_far = []

    for word in list_of_words:
        # remove stopwords
        if word not in stopwords_and_redundancies.STOPWORDS:
            non_common_words_so_far.append(word)

    return non_common_words_so_far


def trim_max_values(full_dict: Dict[str, int], trim_value: int) -> Dict[str, int]:
    """
    This function will mutate the dictionary input by popping out
    entries with the highest key values.
    It removes as many entires as the trim_value input indicates.

    Preconditions:
     - trim_value >= 0
     - len(full_dict) > trim_value

    >>> trimmed = trim_max_values({'renewable': 2, 'actonclimate': 1, 'hello': 5}, 1)
    >>> trimmed == {'actonclimate': 1, 'renewable': 2}
    True
    """
    # change the value into a list
    list_of_values = list(full_dict.values())
    # change the keys into a list.
    list_of_keys = list(full_dict.keys())

    x = trim_value

    while x > 0:
        # gets rid of the most common words, similar to get_max_pairs function
        maximum = max(list_of_values)
        corresponding_key = list_of_keys[list_of_values.index(maximum)]

        full_dict.pop(corresponding_key)
        list_of_keys.remove(corresponding_key)
        list_of_values.remove(maximum)

        # making sure we get the number of dicts we want
        x -= 1

    return full_dict


def get_tweet_vader_score(tweets: List[reading_data.Tweet]) -> List:
    """Given the tweets, return vader score of a sub set of 1000 tweets, scored according
    to VADER lexicon algorithmic computations. Also return the tweet with the
    max score, and min score in the sub set.

    Preconditions:
      - len(tweets) >= 6000
    """
    # attains a sub-set of tweets that is feasibly computable
    condensed_tweets = [tweets[x] for x in range(5000, 6001)]
    # vader scores accumulator
    scores_so_far = []

    for twit in condensed_tweets:
        score_dict = vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer()\
            .polarity_scores(twit.tweet)
        scores_so_far.append(score_dict['compound'])
    # attains tweet with highest vader score and tweet with lowest vader score
    max_score_tweet = condensed_tweets[scores_so_far.index(max(scores_so_far))]
    min_score_tweet = condensed_tweets[scores_so_far.index(min(scores_so_far))]

    return [scores_so_far, max_score_tweet, min_score_tweet]


def get_vader_score_ratios(vader_scores: List[List]) -> List[float]:
    """Returns a list of given vader_scores split into 7 categories, as
    described in vader_piechart

    >>> scores = [[0.9, 0, 0, -0.9], 0, 0]
    >>> get_vader_score_ratios(scores)
    [1, 0, 0, 0, 0, 1, 2]
    """
    # accumulator for positivity ratios
    vader_ratio = [0, 0, 0, 0, 0, 0, 0]

    for score in vader_scores[0]:
        # counts each vader score for each positivity category,
        # from extremely positive to extremely negative
        if score >= 0.85:
            vader_ratio[0] += 1
        elif score >= 0.45:
            vader_ratio[1] += 1
        elif score >= 0.25:
            vader_ratio[2] += 1
        elif score >= -0.25:
            vader_ratio[6] += 1
        elif score >= -0.45:
            vader_ratio[3] += 1
        elif score >= -0.85:
            vader_ratio[4] += 1
        else:
            vader_ratio[5] += 1

    return vader_ratio


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['python_ta.contracts', 'typing', 'reading_data',
                          'vaderSentiment.vaderSentiment', 'vaderSentiment',
                          'stopwords_and_redundancies'],
        'max-line-length': 150,
        'disable': ['R1702', 'R1705', 'C0200']
    })

    import doctest

    doctest.testmod()
