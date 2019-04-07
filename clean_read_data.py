import pandas as pd
import numpy as np

def read_data(path):
    '''
    INPUT: path to .cvs datafile, default = Tweets, Retweets otherwise
    OUTPUT: 1. Tweets Pandas Dataframe object
            WITH :
            Only columns selected for the project,
            Time subset: 2014 - 2019
            Only tweets (columns is_retweet = False)
            Only tweets with account_language = en

            2. User Tweets Pandas Dataframe object
            WITH  2 columns user_id and tweet text. 
            Tweet text is grouped by users
            
    '''

    data = pd.read_csv(path)
    # columns subset

    data = data[['tweetid', 'userid', 'user_display_name', 'user_screen_name',
                'user_reported_location', 'user_profile_description',
                'user_profile_url', 'follower_count', 'following_count',
                'account_creation_date', 'account_language', 'tweet_text', 
                'tweet_time', 'is_retweet']]
    # time subset

    data['tweet_time'] = pd.to_datetime(data['tweet_time'])
    start_date = '2014-01-01'
    end_date = '2018-11-06'
    mask = (data['tweet_time'] > start_date) & (data['tweet_time'] <= end_date)
    data_prime = data.loc[mask]

    # tweets subset

    tweets_mask = data_prime['is_retweet'] == False
    tweets = data_prime[tweets_mask]
    
    # retweets subset
    
    retweets_mask = data_prime['is_retweet'] == True
    tweets = data_prime[retweets_mask]

    # English only. Additional cleaning will be performed in tweet_text_cleanining.py

    tweets_english = tweets[ tweets['account_language'] == 'en']
    retweets_english = tweets[ tweets['account_language'] == 'en']

    #2 Making Users_tweets_series

    # Grouping by tweets by users
    users_tweets_series = tweets_english.groupby(['userid']).apply(lambda x:" ".join(x.tweet_text))
    users_retweets_series = retweets_english.groupby(['userid']).apply(lambda x:" ".join(x.tweet_text))
    # Creating new dataset 
    users_tweets = pd.DataFrame( users_tweets_series ).reset_index()
    users_retweets = pd.DataFrame( users_retweets_series ).reset_index()
    # Formatting columns
    users_tweets.columns = ['userid', 'tweet_text']
    users_retweets.columns = ['userid', 'tweet_text']

    
    return tweets_english, users_tweets
    
#     return retweets_english, users_retweets
