import pandas as pd
import numpy as np
import os

def read_data(path, kind = 'tweets', aggregated_by = None):
    '''
    INPUT: path to .cvs datafile, 
        PArameters: default kind is 'tweets', specify 'retweets' otherwise.
        aggregated_by is None by default, specify 'users' if need to get a dataset
        that is grouped by user
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

    #2 Making Users_tweets_series and User Tweet Subset by grouping by by ['userid]

    # Grouping by tweets by users
    users_tweets_series = tweets_english.groupby(['userid']).apply(lambda x:" ".join(x.tweet_text))
    users_retweets_series = retweets_english.groupby(['userid']).apply(lambda x:" ".join(x.tweet_text))
    
    # Creating new dataset 
    users_tweets = pd.DataFrame( users_tweets_series ).reset_index()
    users_retweets = pd.DataFrame( users_retweets_series ).reset_index()
    
    # Formatting columns
    users_tweets.columns = ['userid', 'tweet_text']
    users_retweets.columns = ['userid', 'tweet_text']
    if kind == 'tweets':
        if aggregated_by == None:
            return tweets_english
        else:
            return users_tweets
    if kind == 'retweets':
        if aggregated_by == None:  
            return retweets_english
        else: 
            return users_tweets    


#########################################
##### Sugested Flow ####################
if __name__ == "__main__": 
    CURRENT_DIR = os.path.dirname('~/galvanize/Emotion_Detection_Elections/data')
    file_path = os.path.join(CURRENT_DIR, 'data/russia_201901_1_tweets_csv_hashed.csv')
    
    print('Read Data:')
    users_df = read_data(file_path, kind = 'tweets', aggregated_by = 'users')
    print(users_df.head(5))
    
    
    
        

