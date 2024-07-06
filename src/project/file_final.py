from joblib import dump
from joblib import load
import pandas as pd
import pickle
from models import ElasticNet_regression


with open('file_elasticnet.pickle','wb') as f:
    pickle.dump(ElasticNet_regression.elastic_model,f) # type: ignore

