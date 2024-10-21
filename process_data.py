import sys
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath:str, categories_filepath:str)->pd.DataFrame:
    """
    Load messages and categories data from CSV files.

    Params:
    messages_filepath : (str)        Path to the messages CSV file.
    categories_filepath : (str)      Path to the categories CSV file.

    Returns:
    pd.DataFrame                     DataFrames with messages and categories.
    """
    def read_csv(filepath: str) -> pd.DataFrame:
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        return pd.read_csv(filepath)

    df_messages = read_csv(messages_filepath)
    df_categories = read_csv(categories_filepath)

    df_merged = pd.merge(df_messages , df_categories, on='id')
    df_merged.reset_index(drop=True, inplace=True)

    return df_merged


def split_categories(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Split a single category column into multiple binary columns.

    Parameters:
    df : pd.DataFrame    The original DataFrame containing the category column.
    column : str         The name of the column to split.

    Returns:
    pd.DataFrame         Parsed DataFrame with the original column dropped.
    """
    categories = df[column].str.split(';', expand=True)

    category_colnames = categories.iloc[0].apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames

    for col in categories:
        categories[col] = categories[col].str.split('-').str[1].astype(int)
        categories[col] = categories[col].apply(lambda x: 1 if x > 0 else 0)

    df = df.drop(column, axis=1)
    df = pd.concat([df, categories], axis=1)

    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from a DataFrame and reset the index."""
    df_cleaned = df.drop_duplicates()
    df_cleaned.reset_index(drop=True, inplace=True)

    return df_cleaned


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Category splitting and Duplicate dropping type cleaning."""
    
    df_category_clean = split_categories(df, "categories")
    df_duplicate_clean = remove_duplicates(df_category_clean)

    return df_duplicate_clean


def save_data(df:pd.DataFrame, database_filename:str)->None:
    """Save cleaned dataframe into SQlite database."""
    try:
        engine = create_engine(f'sqlite:///{database_filename}.db')
        df.to_sql('disaster_response', engine, index=False)
    except Exception as e:
        print(f"An error occurred while saving to SQLite: {e}")


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
