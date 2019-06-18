import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """load csv files
    input: 
    messages_filepath,categories_fielpath
    they are all csv files
    output:
    pandas dataframe df after merging 
    these two csv files 

    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #merge on id
    df = messages.merge(categories,on='id')

    return df


def clean_data(df):
    """clean_data
    input:
    df pandas dataframe
    output:
    df after clean
    the following process will be applied:
    1. categories column will be split into 36 columns
    2. the orignial categories column will be dropped
    3. remove duplicated rows
    """
    #create a dataframe of 36 individual categories
    categories = df['categories'].str.split(";",expand=True)

    #use the first row to get column name
    row = categories.iloc[0,:].values
    category_colnames = [x.split('-')[0] for x in row]

    #rename the column name
    categories.columns = category_colnames

    #convert the values to numeric values 0 or 1
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])

        categories[column] = pd.to_numeric(categories[column])

    #drop original categories
    df.drop(['categories'],axis=1,inplace=True)

    #concate df and categories
    df = pd.concat([df,categories],axis=1)

    #remove duplicated rows
    df.drop_duplicates(inplace=True)

    return df






def save_data(df, database_filename):
    """save date and load into data base
    input:
    df dataframe
    database_filename string 
    output
    None

    this function will save df into database
    """

    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql("messages_categories",engine,index=False)


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
