## Disaster Response Pipeline
This project applies a machine-learning algorithm to disaster messages and can classify the message to different categories that describe disasters. Was trained over tens of thousands of tweets, news messages and direct reports that created during disasters When running the code, you'll get into a GUI in which you'll be able to see visualizations of the dataset the model is based on and to write your own message and check the category it belongs to.

The repository contains the following files:
1. app - a directory with the run.py file which used to run the GUI and a custom transformers file with a customized functions for the NLP task
2. data - a directory the contains the 2 CSV files of the dataset, the database that was created with the processed dataset and a process file with function that performs data cleaning on raw data
3. models -a directory with the model training functions as well as a pickle file of the trained model
-please note that in the train_classifier.py, in the Machine Learning pipeline there are a few parameters that were commented out so the model will run faster. in order to improve the model, you can uncomment these lines.

In order to train the model you'll need to run the following line:
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

To see the GUI, you'll need to run the following command from the 'app' directory:
`python run.py`


