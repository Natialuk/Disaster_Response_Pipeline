import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''extract 2 csv files and creates a merged pandas dataframe 
    input - files path
    output - pandas dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how='left', on='id')
    
    return df


def clean_data(df):
    '''accept a raw dataframe and output a cleaned dataframe with categories 
    input - dataframe
    output - cleaned dataframe
    '''
    categories = df['categories'].str.split(";", expand = True) 
    categories = categories.rename(columns=categories.iloc[0])

    diction = {}
    for col in categories.columns:
        diction[col] = col.split('-', 1)[0]
    
    categories = categories.rename(columns = diction)
    for col in categories.columns:
        categories[col] = categories[col].map(lambda x: int(x[-1:]))
    
    df = df.drop(columns = 'categories')
    df = pd.concat([df, categories], axis=1, sort=False)
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    '''creating an sqlite database file with an sql table
    input - pd dataframe and the db filepath
    output - sqlite db with an sql table
    '''
    engine = create_engine('sqlite:///data//DisasterResponse.db')
    return df.to_sql('DisasterResponse', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()