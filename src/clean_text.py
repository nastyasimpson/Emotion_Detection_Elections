import preprocessor as p
import re

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

import string


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
    punct.add('🇺🇸')
    punct.add('🇯🇵')
    punct.add('🇰🇷')
    punct.add('🇩🇪')
    punct.add('🇨🇳')
    punct.add('🇫🇷')
    punct.add('🇪🇸')
    punct.add('🇮🇹')
    punct.add('🇷🇺')
    punct.add('🇬🇧')
    punct.add('🤗')



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

