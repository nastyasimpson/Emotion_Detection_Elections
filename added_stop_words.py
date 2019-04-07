'''
List of Stop words that is being updated. Words added to the list as not adding interpretability to topic modeling.
Decisions were made using Google Trends Engine. If the word does not show significant fluctuation overtime, 
word considered low impact on opinions.   
'''

new_stop_words = {'rt', 'via', 'new', 'time', 'today', 'one', 'say', 'get', 'go', 
                      'im', 'know', 'need', 'made', 'https', 'http', 'that', 'would', 
                      'take', 'your', 'two', 'yes', 'back', 'look', 'see', 'amp', 'tell',
                      'give', 'httpst', 'htt', 'use', 'dont', 'thing', 'man', 'thank', 'lol', 'cc', 'didnt',
                      'hey', 'like', 'ask', 'let', 'even', 'also', 'ok', 'etc', 'thank', 'ive', 
                     'hi', 'wasnt'}