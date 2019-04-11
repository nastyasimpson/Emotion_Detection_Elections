import pandas as pd  
def get_en_subset(path_to_csv):
    '''
    INPUT: data file
    OUTPUT: subsets of tweets in English, retweets in English

    '''
    data = pd.read_csv(path_to_csv)

    ### Retweets

    retweets_mask = data['is_retweet'] == True
    retweets = data[retweets_mask]
    
    # Tweets
    tweets_mask = data['is_retweet'] == False
    tweets = data[tweets_mask]
    

    #English Only
    tweets_english = tweets[ tweets['account_language'] == 'en']
    retweets_english = retweets[ retweets['account_language'] == 'en']

    return tweets_english, retweets_english