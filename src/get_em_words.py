import pandas as pd
def get_words(data, n = 1104):
    '''
    Input pd DataFrame (Emotions dataset), 
          n -- INT, slise size
    OUTPUT list of words of length n
    '''
    em_data = pd.read_csv(data)
    em_data['word'] = em_data['word'].apply(lambda x: x[:-1])
    word = em_data['word'].tolist()
    words_list = word[:n]
    return words_list



if __name__ == "__main__":  
    data = 'data/Andbrain_DataSet4.csv'