from collections import Counter
''' INPUT: bow --> list of string, 
           words_list -- > list of strings,
    OUTPUT: words_freq -- > dictionary with keys = the words from word_list, 
                                            values = counts of w in bow. 
    WHY: Helper function that counts times of the word is in the bow 
'''
def get_em_words_freq(bow, words_list):
    words_freq = Counter()
    for tweet in bow:
        for word in tweet:
            if word in words_list:
                words_freq[word]  += 1
    return words_freq 



### SUGGESTED WORK FLOW
# get_em_words_freq(bow, words_list)   