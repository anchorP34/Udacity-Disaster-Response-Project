# Udacity-Disaster-Response-Project
Udacity Data Science Nanodegree natural language processing to predict types of disaster messages

The way to run to run this project:

1. Run process_data.py to export data to a database file
  - Need to give locations of messages and 
2. Run train.py to export machine learning model to pickle file
  - Need to give location of database file from process_data output
3. Run run.py to get web application 
  - Will need to point location of pickle file from train.py output
  - Open separate terminal window and run env|grep WORK
  - From the output of that, put in the URL that will replace https://SPACEID-3001.SPACEDOMAIN

## Overview
This project is for the Data Engineering section in the Udacity Data Science Nanodegree program. The bassis of this project is to look at data provided by FigureEight. The data are tweets and texts that were sent during real world disasters and can be labeled into at least one of 36 categories. 

This project required 3 steps:
  1. Create ETL of data from CSV files and upload cleansed data to a database
  2. Write machine learning algorithm to analyze messages and optomize model to correctly classify labels for that text
  3. Create web aplication that can show 3 graphs of overviews of the messages, as well as a input bar that could read a message and correctly classify what label it would belong to.
  
  
## ETL Section (process_data.py)

The ETL secition is your typical extract - transform - load process. Data was provided in CSV's that needed to be read into pandas dataframes, merged together, and then cleaned. 

This involved going through the categories that a message could be labeled as and finding if it was flagged or not. The end dataframe looked like the transcribed message into English, the genre of the text, and the 36 fields that the message could be flagged as.The last piece of cleansing that needed to happen was to eliminate any duplicates.

That pandas Dataframe was then loaded to a SQLite database (hosted by Udacity) to be loaded by the next step of the project.

## Machine Lerning Pipelines (train.py)

After loading the data into a pandas dataframe, I broke the data up into the feature columns and the response columns. This is not a typical machine learning project since there are 36 different response variables. 

I wanted to make my own feature column to be included in my pipeline, so I created my own PopularWords class. The PopularWords class needs to first have an input of a dictionary that shows the word and word count of each word in the messages that are recieved. The other parameter for the class is a percentage, which is used for finding the top X% of words (ordered by frequency) from the dictionary. the logic for this class as a feature becomes "If a message has a lot of common text, it will probably fall into a certian category, while a message with very little common text as other messages is very unique and will probably be flagged with other responses". 

I then create a data pipeline to use natural language processing to clean the words, use token vectorization, and outupt an array of values that compare words in the message to all the total words from all messages. With this and the PopularWords feature, this was the feature set for the training data. 

Since there are 36 response outputs instead of just 1, I needed to use a MultiOutputClassifier for my classification model, with RandomForrests as the core of it. I used a gridsearch of .001, .01 and .1 for the pct hyperparameter of the PopularWords class.

The best model was then saved to a pickle file to be used for future predictions.

## Web Application (run.py)

The web application shows 3 different bar charts: Genre frequency, top 5 response features, and the top 5 words most frequently used in the messages.

If you put in a message into the text bar, you can see what my model would predict of the 36 different feature variables. An easy one to use is "I really need some help!", which gets classified as Related, Request, and Aid Related.


