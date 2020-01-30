import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
nltk.download(['punkt', 'wordnet'])
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


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