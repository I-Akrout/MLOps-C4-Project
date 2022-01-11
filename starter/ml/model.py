from collections import defaultdict
from numpy import test
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV
# Optional: implement hyperparameter tuning.

from .data import process_data
from joblib import load
from json import dump

def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    param_grid = {
        'n_estimators' : [50, 100],
        'max_features' : ['auto', 'sqrt'],
        'max_depth': [20, 50, 100],
        'criterion': ['gini', 'entropy']
    }

    rfc = RFC()
   
    cv_rfc = GridSearchCV(
        estimator=rfc, 
        param_grid=param_grid,
        cv=5
        )

    cv_rfc.fit(X_train, y_train)

    return cv_rfc.best_estimator_


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)

def slice_performance(categorical_features, test_set):
    """
    Performance the slice performance calculation on all
    the categorical features. Saves the results on a files

    Inputs
    ------
    model : ???
        Trained machine learning model.
    categorical_features: list
        list of str containing the categorical features.
    test_set: Pandas Dataframe
        testing set
    """
    model = load('./ml/models/rfc_model.pkl')
    encoder = load('./ml/models/rfc_encoder.joblib')
    lb = load('./ml/models/rfc_lb.joblib')

    output_json = defaultdict(lambda: [])
    for cat in categorical_features:
        for cat_value in test_set[cat].unique():
            tmp_df = test_set[test_set[cat]==cat_value]
            X_test, y_test, _ , _ = process_data(
            tmp_df, 
            categorical_features=categorical_features, 
            encoder=encoder, 
            label="salary",
            training=False,
            lb=lb
        )
    
            y_pred = model.predict(X_test)

            precision, recall, fbeta = compute_model_metrics(
            y_test,
            y_pred
            )
            output_json[cat].append({
                "value":cat_value,
                "precision": precision,
                "recall": recall,
                "fbeta": fbeta
            })
    
    with open('slice_performance.json','w') as fp: 
        dump(output_json,fp)