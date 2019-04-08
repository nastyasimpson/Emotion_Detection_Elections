import pandas as pd
def clean_loc(df):
    '''
    INPUT: padnas dataframe's location columns to edit. 
    OUTPUT: cleaned edited pandas dataframe location columns
    WHY: Clean Up Cities/Countries
    '''
    # US cities
    df = df.replace('Wichita, Kansas', 'Wichita, Kansas, US')
    df = df.replace('NY', 'New York, US')
    df = df.replace('Tampa, FL', 'Tampa, FL, US')
    df = df.replace('Cleveland', 'Cleveland, OH, US')
    df = df.replace('new york', 'New York, US')
    df = df.replace('New York', 'New York, US')
    df = df.replace('New York, NY', 'New York, US')
    df = df.replace('Jersey City, NJ', 'Jersey City, NJ, US')
    df = df.replace('Miami, FL', 'Miami, FL, US')
    df = df.replace('New Jersey, USA', 'New Jersey, US')
    df = df.replace('New York, USA', 'New York, US')

    #Russia
    df = df.replace('Russia, Kaliningrad', 'Kaliningrad, Russia')
    df = df.replace('St. Petersburg, Russia.', 'St. Petersburg, Russia')
    df = df.replace('Sankt-Petersburg', 'St. Petersburg, Russia')

    #England

    df = df.replace('Newcastle', 'Newcastle, England')
    df = df.replace('Chester', 'Chester, England')
    df = df.replace('Lichfield\\t ', 'Lichfield, England')

    # Sweden
    df = df.replace('Stockholm', 'Stockholm, Sweden')

    # France
    df = df.replace('Paris, France', 'Paris, France')
    df = df.replace('Lyon, France', 'Lyon, France')

    # World-wide
    df = df.replace('Earth', 'Worldwide')
    df = df.replace('USA/England/Spain/Itay/Germany', 'Worldwide')
    df = df.replace('Earth', 'Worldwide')
    df = df.replace('World 🌍 *BFF Of SamTheInfidel', 'Worldwide')
    df = df.replace('Earth', 'Worldwide')
    df = df.replace('Istanbul via Liverpool', 'Worldwide')
    df = df.replace('USA  #IslamIsTheProbem #WakeUp', 'Worldwide')
    df = df.replace('United states', 'Worldwide')
    df = df.replace('United States', 'Worldwide')
    df = df.replace('United States 🌎🇺🇸', 'Worldwide')
    df = df.replace('US', 'Worldwide')
    df = df.replace('nan', 'Worldwide')
    df = df.replace('USA', 'Worldwide')
    df = df.replace('USA', 'Worldwide')

    return df

if __name__ == "__main__": 
    data = 'data/russia_201901_1_tweets_csv_hashed.csv'
    data = pd.read_csv(data)
    df = data['user_reported_location']
    print('Cleaning user_reported location:')
    print(df['user_reported_location'].sample(5))