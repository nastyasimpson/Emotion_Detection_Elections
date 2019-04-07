from collections import Counter

def get_em_words_freq(bow, words_list):
    words_freq = Counter()
    for tweet in bow:
        for word in tweet:
            if word in words_list:
                words_freq[word]  += 1
    return(words_freq)  



### SUGGESTED WORK FLOW
# get_em_words_freq(bow, words_list)   