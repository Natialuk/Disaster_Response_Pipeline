## Disaster Response Pipeline
This project applies machine learning algorythm to disaster messages and can classify the message to different catagories that describe disasters.
When running the code, you'll get into a GUI in which you'll be able to see a visualizations of the dataset the model is based on and to write your own message and check the cataegory it's belong to. 

The repository contains the following files:
1. app - a directory with the run.py file which used to run the GUI and a custom transformers file with a customized functions for the NLP task
2. data - a directory the contains the 2 CSV files of the dataset, the database that was created with the processed dataset and a process file with function that preformes data cleaning on raw data
3. models - a directory with the model training functions as well as a pickle file of the trained model 
-please note that in the train_classifier.py, in the Machine Learning pipeline there are a few parameters that where commeted out so the model will run faster.
in order to improve the model, you can uncommet these lines.

In order to train the model you'll need to run the following line:
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

To see the GUI, you'll need to run the following command from the 'app' directory:
`python run.py`


