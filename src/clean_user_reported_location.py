
def clean_user_reportd_location(df):
    '''
    Takes csv_file, reads file into dandas df and cleans user_reported locations. Generalizes country-wide into 
    5 categories: 
    '''
    # read in df file
#     df = pd.read_csv('data')
    # US
    df['user_reported_location'] = df['user_reported_location'].replace('Wichita, Kansas', 'United States')
    df['user_reported_location'] = df['user_reported_location'].replace('United States 🌎🇺🇸', 'United States')
    df['user_reported_location'] = df['user_reported_location'].replace('Cleveland', 'United States')
    df['user_reported_location'] = df['user_reported_location'].replace('NY', 'United States')
    df['user_reported_location'] = df['user_reported_location'].replace('US', 'United States')
    df['user_reported_location'] = df['user_reported_location'].replace('Tampa, FL', 'United States')
    df['user_reported_location'] = df['user_reported_location'].replace('New York, USA', 'United States')
    df['user_reported_location'] = df['user_reported_location'].replace('USA  #IslamIsTheProbem #WakeUp', 'United States')
    df['user_reported_location'] = df['user_reported_location'].replace('United states', 'United States')
    df['user_reported_location'] = df['user_reported_location'].replace('new york', 'United States')
    df['user_reported_location'] = df['user_reported_location'].replace('USA', 'United States')
    df['user_reported_location'] = df['user_reported_location'].replace('Jersey City, NJ', 'United States')
    df['user_reported_location'] = df['user_reported_location'].replace('New York', 'United States')
    df['user_reported_location'] = df['user_reported_location'].replace('New York, NY', 'United States')
    df['user_reported_location'] = df['user_reported_location'].replace('Miami, FL', 'United States')
    df['user_reported_location'] = df['user_reported_location'].replace('New Jersey, USA', 'United States')

    # France
    df['user_reported_location'] = df['user_reported_location'].replace('Paris, France', 'France')
    df['user_reported_location'] = df['user_reported_location'].replace('Lyon, France', 'France')
    
    #Russia
    df['user_reported_location'] = df['user_reported_location'].replace('Russia, Kaliningrad', 'Russia')
    df['user_reported_location'] = df['user_reported_location'].replace('St. Petersburg, Russia.', 'Russia')
    df['user_reported_location'] = df['user_reported_location'].replace('Sankt-Petersburg', 'Russia')

    # World-wide
    df['user_reported_location'] = df['user_reported_location'].replace('Earth', 'Worldwide')
    df['user_reported_location'] = df['user_reported_location'].replace('USA/England/Spain/Itay/Germany', 'Worldwide')
    df['user_reported_location'] = df['user_reported_location'].replace('Earth', 'Worldwide')
    df['user_reported_location'] = df['user_reported_location'].replace('World 🌍 *BFF Of SamTheInfidel', 'Worldwide')
    df['user_reported_location'] = df['user_reported_location'].replace('Earth', 'Worldwide')
    df['user_reported_location'] = df['user_reported_location'].replace('Istanbul via Liverpool', 'Worldwide')
        
    #England
        
    df['user_reported_location'] = df['user_reported_location'].replace('Newcastle', 'England')
    df['user_reported_location'] = df['user_reported_location'].replace('Chester', 'England')
    df['user_reported_location'] = df['user_reported_location'].replace('Lichfield\\t ', 'England')
        
    # Stokholm to Sweden
    df['user_reported_location'] = df['user_reported_location'].replace('Stockholm', 'Sweden')
    # convert nan to Worldwide\n",

    df['user_reported_location'] = df['user_reported_location'].replace('nan', 'Worldwide')

    return df