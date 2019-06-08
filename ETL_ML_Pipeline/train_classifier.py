import sys
from sqlalchemy import create_engine
import pandas as pd
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

#pickle

import pickle


def load_data(database_filepath):
    """read the data base and return dataframe
    input: string the path to sqlite database
    output: dataframe of the data X, Y and categories names
    """
    engine = create_engine(str('sqlite:///')+database_filepath)
    df = pd.read_sql_table("messages_categories",con=engine)

    X = df[df.columns[1]]
    Y = df[df.columns[4:]]
    return X,Y,Y.columns.values




def tokenize(text):
    """
    input: a string
    output word token
    we will perform the following steps:
    1. lower the string
    2. remove punctuation
    3. word_tokenize
    4. lemmatize and remove stop words
    """
    #normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]"," ",text.lower())

    #tokenize text
    tokens = word_tokenize(text)

    #lemmatize and remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words] 

    return tokens



def build_model():
    """
    no input
    build a pipeline of vectorcounts
    estimator
    """
    pipeline = Pipeline([('vect',CountVectorizer(tokenizer=tokenize)),
                         ('tfidf',TfidfTransformer()),
                         ('clf',MultiOutputClassifier(estimator=RandomForestClassifier(n_estimators=50)))
                        ]
                       )
    return pipeline




def evaluate_model(model, X_test, Y_test, category_names):
    """
    input:
    model: sklearn model or pipeline
    X_test,Y_test: input dataframe
    category_names: list of string
    no return value
    """
    y_predict = model.predict(X_test)
    for name in category_names:
        print("category "+name)
        for i in range(Y_test.shape[1]):
            if Y_test.columns[i]==name:
                print(classification_report(Y_test[name],y_predict[:,i]))
                break


def save_model(model, model_filepath):
    """
    input:
    model: pipeline
    model_filepath: string model save path and name
    we will save the model to pickle file
    """
    with open(model_filepath,'wb') as pickle_out:
        pickle.dump(model,pickle_out)




def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
