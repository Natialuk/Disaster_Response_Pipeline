import sys
import pandas as pd
import numpy as np 
import sqlite3
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
import re
from sklearn.multioutput import MultiOutputClassifier
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pickle


def load_data(database_filepath):
    '''loading the data from the DB as SQL, covert it to pd dataframe and split it to X and Y
    input - the database filepath
    output - X, Y datasets + catagory names
    '''
    
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table("DisasterResponse", con=engine)
    X = df['message'].values
    Y = df.drop(columns = ['id', 'message', 'original','genre'])
    
    return X,Y,Y.columns


def tokenize(text):
    '''performs cleaning, tokinization and lemmatization on text
    input - text
    output - cleaned tokens
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class custom_transformers(BaseEstimator, TransformerMixin):
   
    '''a class of customized transformers that can be added to the pipeline'''
    
    def avg_word(self, text):
        
        '''removes numbers and other non-letters and measures the average words' length in each document
        input - text
        output - average words' length
        '''
   
        #take only words and remove numbers and other signs
        text  = "".join((char if char.isalpha() else " ") for char in text)
        # tokenize by words
        words = nltk.word_tokenize(text)

        # measure the average length of words in a sentence
        w = [len(word) for word in words if len(words)!=0]
            
        return sum(w)/len(w) if w else 0
        
    def num_of_stopwords(self, text):
        
        '''counts the number of times a stopword appear in a document
        input - text
        output - count of stopwords per document
        '''
        # tokenize by words
        words = nltk.word_tokenize(text)
        
        # instantiate stopwords dicionary
        stop = stopwords.words('english')
        
        # count the number of stopwords in each sentence
        return len([word for word in words if word in stopwords.words('english')])

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        
        '''performs a transformation on the data and returns a dataframe with scaled avg_word and num_of_stopwords columns
        input - dataset
        output - scaled avg_word and num_of_stopwords columns
        '''
        # apply avg_word function to all values in X
        X_avg = pd.Series(X).apply(self.avg_word)
        # apply num_of_stopwords function to all values in X
        X_stop_words = pd.Series(X).apply(self.num_of_stopwords)
        
        #scale the data to prevent bias
        df_X = pd.concat([X_avg, X_stop_words], axis=1)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_X)
        df_numeric_scaled = pd.DataFrame(scaled_data)
        
        return df_numeric_scaled
    
def build_model():
    
    '''performes machine learning pipeline with gridsearch 
       input - None
       output - a tuned model 
      '''
    pipeline = Pipeline([('features', FeatureUnion([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)), 
            ('tfidf', TfidfTransformer())
        ])),
                
        ('custom', custom_transformers())
        
        ])),
                         
        ('multi', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {
            'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
            #'features__text_pipeline__vect__max_df': (1.0, 2.0),
            #'features__text_pipeline__vect__max_features': (None, 2000, 5000),
            #'features__text_pipeline__tfidf__use_idf': (True, False),
            #'multi__estimator__n_jobs': [1,2,3],
            #'multi__estimator__n_estimators': [10,20],
            #'multi__estimator__min_samples_split': [2, 3, 4],
         }

    cv = GridSearchCV(pipeline, param_grid = parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    
    '''performes prediction and prints classification report for each category in the dataset
       input - model, X_test, Y_test, category_names
       output - precision, recall, and f1-score for each category
     '''     
    y_pred = model.predict(X_test)
    for category, column_pred, column_test in zip(category_names, y_pred, Y_test.values):
        print('classification report for:', category, '\n', classification_report(column_pred, column_test))


def save_model(model, model_filepath):
        
    '''saves the trained model as a pickle file
       input - model, model_filepath
       output - the trained model as a pickle file
     '''  
    temp_pickle = open(model_filepath, 'wb')
    pickle.dump(model, temp_pickle)
    temp_pickle.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Intantiating custom_transformers...')
        custom_transformers()
        
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
