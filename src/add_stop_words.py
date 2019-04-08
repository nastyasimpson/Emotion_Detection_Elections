def add_stop_words(new_word):
    '''
    INPUT: STR, new word to list to the added stop list
    OUTPUT: SET of updated stop words
    '''
    added_stop_words = {'rt', 'via', 'new', 'time', 'today', 'one', 'say', 'get', 'go', 
                        'im', 'know', 'need', 'made', 'https', 'http', 'that', 'would', 
                        'take', 'your', 'two', 'yes', 'back', 'look', 'see', 'amp', 'tell',
                        'give', 'httpst', 'htt', 'use', 'dont', 'thing', 'man', 'thank', 'lol', 'cc', 'didnt',
                        'hey', 'like', 'ask', 'let', 'even', 'also', 'ok', 'etc', 'thank', 'ive', 'hi', 'wasnt'}

    added_stop_words.update(new_word)  
    return added_stop_words                    