import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint

## Imports for Tweet Text Cleaning Pipeline
import preprocessor as p
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import string

### WordCloud
from wordcloud import WordCloud

# LDA Implementation by Gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

# LDA visualizations
import pyLDAvis.gensim

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

####################################################
#####DATA CLEANING AND HANDLING#####################
####################################################

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
   
    return users_tweets    
   

############################################################
############### LDA MODEL ##################################
############################################################
def lda_model(bow):
    '''
    INPUT:
    LST: bag of words (output of read_data.py)
    OUTPUT:
    '''
    # Create Dictionary
    # <gensim.corpora.dictionary.Dictionary at 0x1a550e2518>
    id2word = corpora.Dictionary(bow)
    # Create Corpus
    texts = bow
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # LDA Model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=7, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

    ### Computing Coherence Score for the Model ###
    coherence_model_lda = CoherenceModel(model=lda_model, texts=bow, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
    print('\nCoherence Score: ', coherence_lda) 
    return  (pprint(lda_model.print_topics())),   


##################################################################################################
############## Computing Coherence Values to determine optimal numebr of topics ##################
##################################################################################################

def compute_coherence_values(dictionary, corpus, texts, limit = 10, start=2, step=2):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def plot_coherent_scores(limit =30, start = 1, step = 1, sv = False ):
    
    x = range(start, limit, step)

    plt.figure(figsize=(10,4))

    sns.set_style("darkgrid")
    coherence_values = compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3)[1]
    plt.plot(x, coherence_values, c = 'limegreen')
    plt.xlabel("Num Topics", fontsize = 16)
    plt.ylabel("Coherence score", fontsize = 16)


    plt.xticks((5, 7, 10, 15, 20, 25, 30), fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.grid(c = 'lightcyan')

    plt.axvline(x=7, c = 'darkslateblue', linewidth = 3.0)

    plt.title('Coherence Scores to Determine Optimal No. of Topics ', fontsize = 18)
    if sv == True:
        plt.savefig('coherence_scores.png')
    plt.show()

def plot_word_cloud(cleaned_docs, sv = False):
    all_words = ' '.join([text for text in cleaned_docs])

    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=200, stopwords = ['hey', 'lol', '[][]', 'cc', 'anyon', 'say', 'etc']).generate(all_words)

    plt.figure(figsize=(16, 16))
    plt.imshow(wordcloud, interpolation="bilinear", cmap = 'inferno')
    plt.axis('off')
    plt.title('Unfiltered Words')
    if sv == True:
        plt.savefig('word_cloud.png') 
    plt.show()
       
######################################################################
############### CLEAN TWEET TEXT PIPELINE ############################
######################################################################


def clean_tweet_text(text_column):
    '''
    INPUT: 
    Pandas dataframe column w/tweets text: 
    OUTPUT: 
    Cleaned Docs, bag of words (bow)
    symbol_set charachters removed,
    specified stop words removed
    punctuation removed
    words stemmed and lemmatized
    non-english words removed
    words <3 chars removed
    '''
    punct = set(string.punctuation)
    punct.remove('@')
    punct.remove('#')
    
    ### English stop words from NLTK   
    stop_words = set(nltk.corpus.stopwords.words('english')) 

    ### Custom stop words. 
    added_stop_words = {'rt', 'via', 'new', 'time', 'today', 'one', 'say', 'get', 'go', 'im', 'know', 'need', 'made', 'https', 'http', 'that', 'would', 'take', 'your', 'two', 'yes', 'back', 'look', 'see', 'amp', 'tell', 'give', 'httpst', 'htt', 'use', 'dont', 'thing', 'man', 'thank', 'lol', 'cc', 'didnt', 'hey', 'like', 'ask', 'let', 'even', 'also', 'ok', 'etc', 'thank', 'ive', 'hi', 'wasnt'}

    ### stemmer and lemmer

    lemmer = WordNetLemmatizer()
    stemmer = SnowballStemmer('english')
    
    # converting from pd to list
    corpus = text_column.values.tolist()
    
    #Removing all HTTPs
    docs_no_http = [ re.sub(r'https?:\/\/.*\/\w*', '', doc) for doc in corpus ]
    #First ---> tokenize docs
    tokenized_docs = [doc.split() for doc in docs_no_http]
    # Lower case words in doc
    tokenized_docs_lowered  = [[word.lower() for word in doc]
                                for doc in tokenized_docs]

    # Removing punctuation from docs
    docs_no_punct = [[remove_symbols(word, punct) for word in doc] 
                    for doc in tokenized_docs_lowered]

    # Removing added stop words    
    docs_no_stops1 = [[word for word in doc if word not in added_stop_words] 
                     for doc in docs_no_punct]
    # Removing nltkstop words
    docs_no_stops = [[word for word in doc if word not in stop_words ] 
                     for doc in docs_no_stops1]
    # Lemmatizing words in docs
    docs_lemmatized = [[lemmer.lemmatize(word) for word in doc]
                      for doc in docs_no_stops]
    
    # Stemming words in docs
    docs_stemmed = [[stemmer.stem(word) for word in doc]
                      for doc in docs_lemmatized]
    
    # Removes mentions, emotions, hashtags and emojies
    docs_no_mentions = [preprocessing_text(' '.join(doc)) for doc in docs_stemmed]
    
    # Removes all non-english charachters and any other different charachters
    docs_english_only = [re.sub(r'[^a-zA-Z]', " ", doc) for doc in docs_no_mentions]
    
    # keeping words that are more than 2 chars long
    cleaned_docs = []
    for doc in docs_english_only:
        cleaned_docs.append(' '.join(word for word in doc.split() if len(word)>2))
   
    # converting cleaned docs into list of lists i.e bag of words (bow) per each doc
    bow = [list(tweet.split(' ')) for tweet in cleaned_docs]

    return cleaned_docs, bow 

def preprocessing_text(text):
    '''
    INPUT: str
    OUTPUT: str w/ emojies, urls, hashtags and mentions removed
    '''
    p.set_options(p.OPT.EMOJI, p.OPT.URL, p.OPT.HASHTAG, p.OPT.MENTION, p.OPT.NUMBER,
    p.OPT.RESERVED, p.OPT.SMILEY)
    clean_text = p.clean(text)
    return clean_text

def remove_symbols(word, symbol_set):

    '''
    INPUT: STR,   symbol set (i.e. punctuation)
    OUTPUT:word w/ removed symbols from specified symbol set
    '''
    return ''.join(char for char in word 
                    if char not in symbol_set)



######################################################################
##### Main function with lots of commented out code for workflow:#####
######################################################################
if __name__ == "__main__": 
    data = 'data/russia_201901_1_tweets_csv_hashed.csv'
    print('Read Data:')
    users_df = read_data(data)
    print('Cleaned Data: ')
    text_column = users_df['tweet_text']
    print('Ready to Clean the Tweets!')
    cleaned_docs, bow = clean_tweet_text(text_column)
    print('The Tweets are Clean: ')
    print('Model is ready! \nBelow you can see 7 topics and most frequent keywords: ')
    lda_model(bow)
    