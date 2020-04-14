import sys

import numpy as np
import pandas as pd

from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    """Creates concatenated dataframe from proviced csv files paths"""
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = pd.concat([messages, categories], axis=1)

    return df


def clean_data(df):
    """Performs cleaning operations on provided dataframe by
    1. transforming categories column into multiple columns for every category with bool value
    2. dropping old categories column from df
    3. concatenating df and created one hot categories dataframe
    """
    categories_df = df['categories'].str.split(';', expand=True)
    # extract category name from strings like "category-0"
    categories_names = categories_df.loc[0].apply(
        lambda category: category.split('-')[0]
        )

    categories_df.columns = categories_names

    for column in categories_names:
        # set each value to be the last character of the string
        categories_df[column] = categories_df[column].str[-1]
        categories_df[column] = pd.to_numeric(categories_df[column])

    df.drop("categories", axis=1, inplace=True)
    df = pd.concat([df, categories_df], axis=1)

    df = df.drop_duplicates()

    return df


def save_data(df, database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('DisasterResponse', engine, index=False)


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