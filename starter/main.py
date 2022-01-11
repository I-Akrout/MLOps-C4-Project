# Put the code for your API here
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
from joblib import load
from starter.ml.data import process_data
from starter.ml.model import inference
import numpy as np
import pandas as pd


class DataEntry(BaseModel):
    age: int
    fnlgt: int
    educationNum: int
    capitalGain: int
    capitalLoss: int
    hoursPerWeek: int
    workclass:Literal[
        'State-gov', 
        'Self-emp-not-inc', 
        'Private', 
        'Federal-gov',
        'Local-gov', 
        'Self-emp-inc', 
        'Without-pay'
    ]
    education: Literal[
        'Bachelors', 
        'HS-grad', 
        'Doctorate', 
        'Assoc-voc', 
        'Assoc-acdm', 
        '11th', 
        'Masters', 
        '9th',
        'Prof-school',
        '5th-6th', 
        'Some-college',
        '7th-8th', 
        '10th', 
        'Preschool', 
        '12th', 
        '1st-4th'
    ]
    maritalStatus: Literal[
        'Never-married', 
        'Married-spouse-absent', 
        'Separated', 
        'Married-AF-spouse',
        'Widowed',
        'Married-civ-spouse', 
        'Divorced',
    ]
    occupation: Literal[
        'Adm-clerical', 
        'Exec-managerial', 
        'Handlers-cleaners',
        'Machine-op-inspct', 
        'Tech-support',
        'Craft-repair', 
        'Protective-serv', 
        'Armed-Forces',
        'Priv-house-serv',
        'Prof-specialty', 
        'Other-service', 
        'Sales', 
        'Transport-moving',
        'Farming-fishing'
    ]
    relationship: Literal[
        'Not-in-family', 
        'Husband', 
        'Unmarried', 
        'Own-child',
        'Other-relative',
        'Wife'
    ]
    race: Literal[
        'White', 
        'Black', 
        'Asian-Pac-Islander', 
        'Amer-Indian-Eskimo',
        'Other'
    ]
    sex: Literal[
        'Male', 
        'Female'
    ]
    nativeCountry: Literal[
        'United-States', 'Cuba', 'Jamaica', 
        'India', 'Mexico','Puerto-Rico', 
        'Honduras', 'England', 'Canada', 
        'Germany', 'Iran','Philippines', 
        'Poland', 'Columbia', 'Cambodia', 
        'Thailand','Ecuador', 'Laos', 
        'Taiwan', 'Haiti', 'Portugal',
        'Dominican-Republic', 'El-Salvador', 
        'France', 'Guatemala','Italy', 
        'China', 'South', 'Japan', 
        'Yugoslavia', 'Peru','Outlying-US(Guam-USVI-etc)', 
        'Scotland', 'Trinadad&Tobago','Greece', 
        'Nicaragua', 'Vietnam', 'Hong', 
        'Ireland', 'Hungary','Holand-Netherlands'
    ]

api = FastAPI()


@api.get("/")
async def message():
    return {
        "message":"Welcome to salary predictor"
    }

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

@api.post("/infer")
async def infer(data: DataEntry):
    model = load('./starter/ml/models/rfc_model.pkl')
    lb = load('./starter/ml/models/rfc_lb.joblib')
    encoder = load('./starter/ml/models/rfc_encoder.joblib')

    data_entry  = np.array([[
                     data.age,
                     data.fnlgt,
                     data.educationNum,
                     data.capitalGain,
                     data.capitalLoss,
                     data.workclass,
                     data.education,
                     data.maritalStatus,
                     data.occupation,
                     data.relationship,
                     data.race,
                     data.sex,
                     data.hoursPerWeek,
                     data.nativeCountry
                     ]])

    data_df = pd.DataFrame(data=data_entry, columns=[
        "age",
        'fnlgt',
        'educationNum',
        'capitalGain',
        'capitalLoss',
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "hours-per-week",
        "native-country"
    ])
    
    X, _, _, _ = process_data(
                data_df,
                categorical_features=cat_features,
                encoder=encoder, 
                lb=lb, 
                training=False
            )
    pred = inference(model, X)
    return {"prediction":lb.inverse_transform(pred)[0]}