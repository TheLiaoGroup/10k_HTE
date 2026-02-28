from model.SVR import SVR
from model.RandomForest import RF
from model.Ridge import RidgeRegression
from model.Lasso import LassoRegression
from model.GBM import GBM
from model.XGBoost import XGBoost

model_mapping = {
    "SVR": SVR,
    "RandomForest": RF,
    "Ridge": RidgeRegression,
    "Lasso": LassoRegression,
    "GBM":GBM,
    "XGBoost":XGBoost
}

def model_dispatcher(func):
    def wrapper(model_type, df, sub1_column, sub2_column,condition_id_column,product_column):
        if model_type in model_mapping:
            model_mapping[model_type](df, sub1_column, sub2_column,condition_id_column, product_column)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    return wrapper

@model_dispatcher
def train_machine_learning_model(model_type, df, sub1_column, sub2_column,condition_id_column,product_column):
    pass 



