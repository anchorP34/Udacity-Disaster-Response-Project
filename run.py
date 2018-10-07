import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin


##################################################
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
##################################################

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
        """
        Returns self, but is needed for pipeline.
        """
        return self
            
    
    def transform(self, X):
        """
        Transforms messages to find the number of words that are most common in all messages to determine
        how unique a message is.
        """
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



app = Flask(__name__)

def tokenize(text):
    """
    Cleans messages for preparation for pipelines
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
# Additions by Payton

database_filepath = 'data/DisasterResponse.db'
file_path = 'sqlite:///'+ database_filepath

engine = create_engine(file_path)
df = pd.read_sql_table('StagingMLTable', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Custom Viz 1
    viz_1 = pd.DataFrame(df.iloc[:,4:].mean(), columns = ['Val']).sort_values('Val', ascending = False).iloc[:5,:]
    top_5_vals = viz_1['Val']
    top_5_cols = list(viz_1.index)
    
    # Custom Viz 2 Setup
    popular_words = {}

    stop_words = stopwords.words('english')
    punct = [p for p in string.punctuation]


    for m in df['message']:
        for word in m.split():
            new_word = word.lower()
            if new_word not in stop_words and new_word not in punct:
                if new_word in popular_words:
                    popular_words[new_word] += 1
                else:
                    popular_words[new_word] = 1
    
    viz_2 = pd.DataFrame.from_dict(popular_words, orient = 'index')
    viz_2.columns = ['Val']
    top_5_words_vals = viz_2.sort_values('Val', ascending = False)[:5]['Val']
    top_5_words = list(viz_2.sort_values('Val', ascending = False)[:5].index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
     , {
            'data': [
                Bar(
                    x=top_5_cols,
                    y=top_5_vals
                )
            ],

            'layout': {
                'title': 'Top 5 Message Types',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Message Type"
                }
            }
        }
        , {
            'data': [
                Bar(
                    x=top_5_words,
                    y=top_5_words_vals
                )
            ],

            'layout': {
                'title': 'Top 5 Most Used Words',
                'yaxis': {
                    'title': "Total Count from all Messages"
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """
    Create port location to run application on
    """
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
