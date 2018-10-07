# import libraries
import sys
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine

import operator
import string
import re

import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath, table_name = 'StagingMLTable'):
    """
    Load data from database
    
    Input: data_filepath - String
           The file path / connection string to the database
           
           table_name - String
           The table that contains the data of interest
           
    Output: X - Pandas DataFrame
            Feature variables to be used in machine learning
            y - Pandas DataFrame
            Response variables that will be used to predicted
            
            column_names - List
            List of the y response variable column names
            
            popular_words - Dictionary
            Dictionary of all the words from all messages and their word frequencies
    """
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table(table_name,engine)
    X = df.loc[:, 'message']
    y = df.iloc[:, 4:]
    
    popular_words = {}
    
    # Want to exclude stop words and punctuation
    stop_words = stopwords.words('english')
    punct = [p for p in string.punctuation]

    # Load the popular_words dicitonary with the unique words in all the messages and the word frequency of them
    for m in df['message']:
        for word in m.split():
            new_word = word.lower()
            if new_word not in stop_words and new_word not in punct:
                if new_word in popular_words:
                    popular_words[new_word] += 1 # Add to already existing value
                else:
                    popular_words[new_word] = 1 # Create key / value pair in dictionary
    
    return X, y, list(y.columns), popular_words

class PopularWords(BaseEstimator, TransformerMixin):
    def __init__(self, word_dict, pct = .001):
        """
      This class is to see if the input message has common words that other messages have, or
      if it is truly a unique message. The theory is that certain messages can be grouped together
      based on the number of common words that are in the message.
      
      Input: word_dict - Dictionary
             Dictionary of words (key) with the word frequency (value)
             
             pct - Percentage (float)
             The top X% of common words that you would like to compare to in the word_dict.
             The lower the number of words, the more unique the message
             
        """
        self.word_dict = word_dict
        if pct == None:
            self.n = int(len(self.word_dict) * .01)
        else:
            self.n = int(len(self.word_dict) * pct)
        
    def fit(self, X, y = None):
        return self
            
    
    def transform(self, X):
        
        def get_word_count(message_list, top_word_count, sorted_dict):
            """
              Returns the total number of words that are in the message that are in the top X most
              frequent words out of all the messages
              
              Input: message_list - String
                     The messages that are going to be input
                     
                     top_word_count - Int
                     The top number of most frequent words in all messages to compare to individual message
                     
                     sorted_dict - Dictionary
                     The sored dictionary of all the word frequencies
                     
              Output: total_count - Int
                      The total number of words from the message that are in the most frequent number of input words 
                      from all messages
              
            """
            total_count = 0
            for w in range(top_word_count):
                if sorted_dict[w][0] in message_list:
                    total_count +=1
        
            return total_count 
        
        # Sort the dictionary from most frequent words to least frequent words
        sorted_dict = sorted(self.word_dict.items(), key=operator.itemgetter(1), reverse = True)
        
        # Make the words lowercase
        lower_list = pd.Series(X).apply(lambda x: x.lower())
        
        # Get rid of punctuation
        no_punct = pd.Series(lower_list).apply(lambda x: re.sub(r'[^\w\s]','', x))
        
        # Create list of the words that are not stop words
        final_trans = pd.Series(no_punct).apply(lambda x: x.split())
        
        # Get the top number of words that want to be viewed from the dictionary
        top_word_cnt = self.n
        
        # Get the results
        results = pd.Series(final_trans).apply(lambda x: get_word_count(x,top_word_cnt, sorted_dict)).values
        # Put in DataFrame output to work with pipeline
        return pd.DataFrame(results)


def tokenize(text):
    """
       Tokenizer that will clean the text to look at the lowercase words and break it into matrix for machine learning
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(word_dict):
    """
    Build the pipeline model that is going to be used as the model
    Input: word_dict - Dictionary
           The word dictionary from all of the messages
           
    Output: cv - model
            The model structure to be used for fitting and predicting
    """
    popular_words = word_dict
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('pop_words', PopularWords(word_dict = popular_words, pct = .001))
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state = 534)))
    ])
    # Setting up the parameters for the grid search for different values of pct for the pop_words feature     
    parameters = {
        'features__pop_words__pct': [.001, .01 , .1]
        }         
    cv = GridSearchCV(pipeline, param_grid = parameters)
    return cv
    


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evalueate how the model is doing. Prints out the classification report of each response variable
    """
    preds = model.predict(X_test)
    
    for idx, col in enumerate(category_names):
        print(col, classification_report(Y_test.iloc[:,idx], preds[:,idx]))
              


def save_model(model, model_filepath):
    """
      Saves the model to the specified model path
    """
    model_pkl = open(model_filepath, 'wb')
    pickle.dump(model, model_pkl)
    model_pkl.close()


def main():
    """
    Create machine learning models and save output to pickle file
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names, popular_words = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(popular_words, )
        
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
