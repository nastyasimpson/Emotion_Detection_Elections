
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

import string
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

import datetime
datetime.datetime.strptime

import preprocessor as p
import re

# Used to clean twitter data
from clean_read_data import read_data

####################################
#### DATA CLEANING AND HANDLING ####
####################################

## Cleaning up Dataset with Emotion frequencies: 

def clean_up_em_df(em_df):
    '''
    INPUT: pandas dataframe column containing STRING 'word' to be cleaned
    OUTPUT: cleaned string 'word' ready for vectorizing. 
    WHY: Cleaning function for Emotion frequencies Data set
    '''
    # clean up the word column
    em_df["word"] = em_df.word.str.strip().str.lower()
    # remove duplicates
    em_df = em_df.groupby('word').mean()

    return em_df

## Cleaning up and Subsetting Twitter Dataset

def light_clean_tweets(df_column):
    '''
        INPUT: pandas dataframe column containing STRING 'tweets' to be cleaned
        OUTPUT: pandas dataframe column containing STRING 'tweets' with cleaned tweets 
        (lemmatized, lower case, stemmer, remove symbols, reg expressions)
        WHY: Preparing Tweet text for linear transformations. 

    '''
    string.punctuation
    punct = set(string.punctuation)
    punct.remove('@')
    punct.remove('#')
    punct.add('🇺🇸')
    punct.add('🤢')

    lemmer = WordNetLemmatizer()
    stemmer = SnowballStemmer('english')
     

    df_column = df_column.apply(lambda x: x.lower())
    df_column = df_column.apply(lambda x: remove_symbols(x, punct))
    df_column = df_column.apply(lambda x: re.sub(r'https?:\/\/.*\/\w*', '', x))
    df_column = df_column.apply(lambda x: lemmer.lemmatize(x))
    df_column = df_column.apply(lambda x: stemmer.stem(x))
    return df_column 


def remove_symbols(word, symbol_set):
    
    '''
    INPUT: string, symbol set (i.e punctuation or any other symbol)
    OUTPUT: word with any symbols specified in symbol set removed
    WHY: Removing symbols from word. Helper function for light clean tweets
    '''
    return ''.join(char for char in word 
                    if char not in symbol_set)


def df_transform(df_column, em_df):
    ''' INPUT: twitter df with cleaned tweets, cleaned em frequencies df w/ cleaned text
        OUTPUT: data frame with columns as 7 basic emotions, rows as tweets. Each tweets has an output in each emotion
        
    '''
    vectorizer = CountVectorizer(stop_words='english', vocabulary = em_df.index)
    vectorizer.fit(df_column)
    X = vectorizer.transform(df_column)
    if ((em_df.index == vectorizer.get_feature_names()).all() )and (X.shape[1] == em_df.values.shape[0]) :
        em_freq_np = X @ em_df.values
        em_freq_df = pd.DataFrame.from_records(em_freq_np)
        em_freq_df.columns = em_df.columns
        return em_freq_df
    else: 
        print('Mismatching index for dataframes or wrong shape. Unable to multiply')    

def em_time(df_tweet, em_freq_df):
    ''' INPUT: df w/ text and time columns
        OUTPUT: df w/time as an index of df
    '''

    df_tweet.reset_index(drop = True)
    emotions_frequencies = pd.concat([df_tweet, em_freq_df], axis = 1)
    return emotions_frequencies

def monthly_tweets(df):
    # Number of Tweets Over Time
    df_time = pd.DataFrame(pd.to_datetime( df["tweet_time"] ))

    df_time["count"] = 1
    df_time.set_index("tweet_time", inplace = True)

    # monthly
    monthly = df_time['count'].resample('M').sum()
    return monthly


def plot_emotional_composition():
    '''
    Plots Emotional Composition of tweets over time
    '''
    # Dealing w/disgust
    disgust= emotions_frequencies[['tweet_time', 'disgust' ]]
    disgust_time = pd.DataFrame(pd.to_datetime( disgust["tweet_time"] ))
    disgust_time['disgust'] = disgust['disgust']
    disgust_time.set_index("tweet_time", inplace = True)
     # Dealing w/surprise
    surprise= emotions_frequencies[['tweet_time', 'surprise' ]]
    surprise_time = pd.DataFrame(pd.to_datetime( surprise["tweet_time"] ))
    surprise_time['surprise'] = surprise['surprise']
    surprise_time.set_index("tweet_time", inplace = True)
    
    # Neutral
    neutral= emotions_frequencies[['tweet_time', 'neutral' ]]
    neutral_time = pd.DataFrame(pd.to_datetime( neutral["tweet_time"] ))
    neutral_time['neutral'] = neutral['neutral']
    neutral_time.set_index("tweet_time", inplace = True)
    
    # Anger
    anger= emotions_frequencies[['tweet_time', 'anger' ]]
    anger_time = pd.DataFrame(pd.to_datetime( anger["tweet_time"] ))
    anger_time['anger'] = anger['anger']
    anger_time.set_index("tweet_time", inplace = True)
    
    # Sad
    sad= emotions_frequencies[['tweet_time', 'sad' ]]
    sad_time = pd.DataFrame(pd.to_datetime( sad["tweet_time"] ))
    sad_time['sad'] = sad['sad']
    sad_time.set_index("tweet_time", inplace = True)

    # Happy

    happy= emotions_frequencies[['tweet_time', 'happy' ]]
    happy_time = pd.DataFrame(pd.to_datetime( happy["tweet_time"] ))
    happy_time['happy'] = happy['happy']
    happy_time.set_index("tweet_time", inplace = True)
    
    # Fear

    fear= emotions_frequencies[['tweet_time', 'fear' ]]
    fear_time = pd.DataFrame(pd.to_datetime( fear["tweet_time"] ))
    fear_time['fear'] = fear['fear']
    fear_time.set_index("tweet_time", inplace = True)

    monthly_disgust = disgust_time['disgust'].resample('M').sum()
    monthly_surprise = surprise_time['surprise'].resample('M').sum()
    monthly_neutral = neutral_time['neutral'].resample('M').sum()
    monthly_anger = anger_time['anger'].resample('M').sum()
    monthly_sad = sad_time['sad'].resample('M').sum()
    monthly_happy = happy_time['happy'].resample('M').sum()
    monthly_fear = fear_time['fear'].resample('M').sum()
    
    norm_monthly_disgust = disgust_time['disgust'].resample('M').sum()/monthly
    norm_monthly_surprise = surprise_time['surprise'].resample('M').sum()/monthly
    norm_monthly_neutral = neutral_time['neutral'].resample('M').sum()/monthly
    norm_monthly_anger = anger_time['anger'].resample('M').sum()/monthly
    norm_monthly_sad = sad_time['sad'].resample('M').sum()/monthly
    norm_monthly_happy = happy_time['happy'].resample('M').sum()/monthly
    norm_monthly_fear = fear_time['fear'].resample('M').sum()/monthly
    
    plt.figure(figsize=(22,8))
    sns.set(style="darkgrid")
    plt.plot(monthly_disgust,  c = 'mediumorchid', linewidth=2.5, label = 'disgust')

    plt.plot(monthly_surprise,  c = 'cyan', linewidth=2.5, label = 'surprise')
    plt.plot(monthly_neutral,  c = 'lightpink', linewidth=2.5, label = 'neutral')
    plt.plot(monthly_anger,  c = 'darkslateblue', linewidth=2.5, label = 'anger')
    plt.plot(monthly_sad,  c = 'sienna', linewidth=2.5, label = 'sad')
    plt.plot(monthly_happy,  c = 'yellow', linewidth=2.5, label = 'happy')
    plt.plot(monthly_fear,  c = 'tomato', linewidth=2.5, label = 'fear')

    plt.legend(borderpad=2.5, fontsize = 15, loc='upper right')
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.ylabel('Emotion Detection. Tweets ', fontsize = 20)
    plt.grid(c = 'lemonchiffon', linestyle = '--')
    plt.title('Detecting Seven Basic Emotions: 2014 -- 2019 ', fontsize = 24)
    plt.show()
    # plt.savefig('plots/monthly_tweet_volume.png')

    plt.figure(figsize=(22,8))
    sns.set(style="darkgrid")
    plt.plot(norm_monthly_disgust,  c = 'mediumorchid', linewidth=2.5)

    plt.plot(norm_monthly_surprise,  c = 'cyan', linewidth=2.5)
    plt.plot(norm_monthly_neutral,  c = 'lightpink', linewidth=2.5)
    plt.plot(norm_monthly_anger,  c = 'darkslateblue', linewidth=2.5)
    plt.plot(norm_monthly_sad,  c = 'sienna', linewidth=2.5)
    plt.plot(norm_monthly_happy,  c = 'yellow', linewidth=2.5)
    plt.plot(norm_monthly_fear,  c = 'tomato', linewidth=2.5)

    plt.legend(borderpad=2.5, fontsize = 20, loc='upper left')
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.ylabel('Cumulative Emotion Detection. Tweets ', fontsize = 20)
    plt.grid(c = 'lemonchiffon', linestyle = '--')
    plt.title('Detecting Seven Basic Emotions: 2014 -- 2019 \n Normalized ', fontsize = 24)
    # plt.savefig('plots/norm_monthly_tweet_volume.png')

    plt.show()

######################################################################
##### Main function: #################################################
######################################################################
if __name__ == "__main__":
    
    # 1. Read Emotions Dataset: 
    print('Cleaning Emotions Dataset')
    CURRENT_DIR = os.path.dirname('~/galvanize/Emotion_Detection_Elections/data')
    file_path = os.path.join(CURRENT_DIR, 'data/andbrainDataSet108.csv')
    em_data = pd.read_csv(file_path)
    # 2. Clean Emotions Data Set
    em_df = clean_up_em_df(em_data)
    print(em_df.sample())
    
    # Read and Clean Twitter Data

    print('Read Twitter Data:')
    CURRENT_DIR = os.path.dirname('~/galvanize/Emotion_Detection_Elections/data')
    file_path = os.path.join(CURRENT_DIR, 'data/russia_201901_1_tweets_csv_hashed.csv')
    tweets_english = read_data(file_path, kind = 'tweets', aggregated_by = None)
    print(tweets_english.head(5))
    
    # Clean Tweet Text
    print('Cleaning Twitter Text: ')
    # df_column = tweets_english['tweet_text']    
    df_column = light_clean_tweets(tweets_english['tweet_text'])

    # Vectorize and Multiply two data frames to get emotional compositions of each tweet. 
    print('Vectorizing tweets over the set of words with their emotional composition.')
    em_freq_df = df_transform(df_column, em_df)
    print(em_freq_df.sample(3))

    # Concatenate Dataframes to have one dataframe with emotional composition of tweets and tweet time. 
    print('Generating Emotional composition per tweets dataframe')
    emotions_frequencies = em_time(tweets_english[['tweet_text', 'tweet_time']], em_freq_df)
    print(emotions_frequencies.sample(3))

    # Getting Monthly Volume
    print('Getting Monthly volume')
    monthly = monthly_tweets(tweets_english)
    print('Emotional Composition Plots: ')
    plot_emotional_composition()


    
    
  