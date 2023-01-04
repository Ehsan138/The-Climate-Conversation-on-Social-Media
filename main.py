"""CSC110 Fall 2020 Project, Phase 2 The Climate Conversationon Social Media:
Finding Trends in Tweets Related to Climate Change

Description
===============================
This is the main module, use it to execute the code!

Copyright and Usage Information
===============================
This file is provided solely for the personal and private use of the professors and TAs
in CSC110 at the University of Toronto St. George campus. All forms of
distribution of this code, whether as given or with any changes, are
expressly prohibited. For more information on copyright for this CSC110 project,
please consult with us.

This file is Copyright (c) 2020 Jiajin Wu, Ehsan Emadi, Ashkan Aleshams and Michael Galloro.
"""
# required files:
# reading_data file:
import reading_data as rd

# computing_data file:
import computing_data as cd

# visualizing_data file:
import visualizing_data as vd

# Please always keep this section uncommented while running the following examples:
tweets = rd.make_tweets(rd.read_file_data('climate_id.jsonl'))

# Read-me and VADER context
tweets_for_vader = cd.get_tweet_vader_score(tweets)
vd.visualization_context(tweets_for_vader)

####################################################################################
####################################################################################
# Word Cloud Graphs:
####################################################################################
####################################################################################

# EXAMPLE 1: Word cloud for locations:
tweets_temp = cd.accumulate_locations(tweets)
vd.locations_wordcloud(tweets_temp, 'images/earth.png')

# EXAMPLE 2: Word cloud for users' descriptions:
tweets_temp = cd.accumulate_descriptions(tweets)
vd.descriptions_wordcloud(tweets_temp, 'images/brain.png')

# EXAMPLE 3: Word cloud for tweets' words:
tweets_temp = cd.accumulate_tweet_words(tweets)
vd.tweet_words_wordcloud(tweets_temp, 'images/speech.png')

# EXAMPLE 4: Word cloud for hashtags:
tweets_temp = cd.accumulate_hashtags(tweets)
vd.hashtags_wordcloud(tweets_temp, 'images/bird.png')

####################################################################################
####################################################################################
# VADER Pie Chart:
####################################################################################
####################################################################################

# EXAMPLE 5: VADER pie chart:
vd.vader_piechart(tweets_for_vader)
