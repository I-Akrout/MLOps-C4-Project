# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.

import pandas as pd
from ml.data import process_data
from ml.model import *
from ml.clean_data_helper import load_data
import joblib



import logging
logging.basicConfig(
    filename='./train_logs/census_train.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S"
)



# Add code to load in the data.
logging.info('INFO: Loading data.')
data = load_data('./census_clean.csv') 
logging.info('SUCCESS: Data Loaded')

# Optional enhancement, use K-fold cross validation
# instead of a train-test split.
logging.info('INFO: Splitting data')
train, test = train_test_split(data, test_size=0.20)
logging.info('SUCCESS: Data Splitted')

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

logging.info('INFO: Processing training data ...')
X_train, y_train, encoder, lb = process_data(
    train, 
    categorical_features=cat_features, 
    label="salary", 
    training=True
)
logging.info('SUCCESS: Training data processed')
logging.info(f'INFO: X_train.shape: {X_train.shape}')
logging.info(f'INFO: y_train.shape: {y_train.shape}')

logging.info('INFO: Processing testing data ...')
# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, 
    categorical_features=cat_features, 
    encoder=encoder, 
    label="salary",
    training=False,
    lb=lb
)
logging.info('SUCCESS: Testing data processed')
logging.info(f'INFO: X_test.shape: {X_test.shape}')
logging.info(f'INFO: y_test.shape: {y_test.shape}')


# Train and save a model.
logging.info('INFO: Starting training process ...')
model = train_model(X_train, y_train)

precision, recall, fbeta = compute_model_metrics(
    y_train, 
    inference(
        model, 
        X_train
        )
    )
logging.info(f'INFO: Train metrics: Precision: \
    {precision}, recall: {recall}, fbeta: {fbeta}')



precision, recall, fbeta = compute_model_metrics(
    y_test, 
    inference(
        model, 
        X_test
        )
    )
logging.info(f'INFO: Train metrics: Precision:\
    {precision}, recall: {recall}, fbeta: {fbeta}')

logging.info('INFO: Saving the model ...')
joblib.dump(model, './ml/models/rfc_model.pkl')
logging.info('SUCCESS: Model saved')

logging.info('INFO: Saving the data encoder ...')
joblib.dump(encoder, './ml/models/rfc_encoder.joblib')
logging.info('SUCCESS: Encoder saved')

logging.info('INFO: Saving the data LB ...')
joblib.dump(lb, './ml/models/rfc_lb.joblib')
logging.info('SUCCESS: LB saved')

slice_performance(cat_features, test)

