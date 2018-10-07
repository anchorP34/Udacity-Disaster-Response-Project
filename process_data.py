# import libraries
import sys
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine

# Load in the messages and the affiliated categories
def load_data(messages_filepath, categories_filepath):
    """
    Load data from a database
    Input: messages_filepath - String
            The path to the messagages data csv
           categories_filepath - String
            The path to the categories data csv
    Output: df = DataFrame
            Merged DataFrame of message and categories data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # Merge the two datasets together
    df = pd.merge(messages, categories)
    
    return df


def clean_data(df):
    """
    Extracts categories and flags from categories data, remove duplicates
    Input: df - DataFrame
            Dataframe output from load_data function
    Output: df - DataFrame
            Cleansed dataframe of the input data
    """
    # Get the categories column from the DF, then find the strings attached
    categories = df['categories'].str.split(n = 36, pat = ';', expand = True)
    row = categories.iloc[0,:]
    # Strip the value before the "-"
    category_colnames = [val.split('-')[0] for val in row.values]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # Drop the old categories columns
    df.drop('categories', axis= 1, inplace =True)
    
    # Add in the new categories columns to the end of the dataframe
    df = pd.concat([df, categories], axis=1, join_axes=[df.index])
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    return df


def save_data(df, database_filename, table_name = 'StagingMLTable'):
    """
    Save data to database
    Input: df - DataFrame
            DataFrame from clean_data dataframe
           database_filename - String
           Database file location of where data is to be stored
           table_name - String (default is StagingMLTable)
           The name of the table that the data should be saved to 
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(table_name, engine, index=False)  
    print("Data was saved to {} in the {} table".format(database_filename, table_name))


def main():
    """
    Run ETL of messages and categories data
    """
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
