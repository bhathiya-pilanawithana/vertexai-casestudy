
import argparse
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from hypertune import HyperTune
import os
import logging

data_location="gs://quixotic-galaxy-439523-m3-buckt/boston_housing.csv"

logging.getLogger().setLevel(logging.INFO)

def train_model(data, n_estimators, max_depth, learning_rate, subsample):
    
    y = data.pop("MEDV")
    X = data
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

    params = {
        'objective': 'reg:squarederror',
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': subsample
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    
    mmse_val = mean_squared_error(y_val, model.predict(X_val))

    hpt = HyperTune()
    #hpt.report_hyperparameter_tuning_metric(
    #    hyperparameter_metric_tag='mmse_val',
    #    metric_value=mmse_val,
    #    global_step=1000)
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='mmse_val',
        metric_value=mmse_val)

    return model

def get_args():
    parser = argparse.ArgumentParser(description='XGBoost Hyperparameter Tuning')
    parser.add_argument('--n_estimators', type=int, default=10)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--subsample', type=float, default=0.1)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    data = pd.read_csv(data_location)
    model = train_model(data=data, 
                        n_estimators=args.n_estimators, 
                        max_depth=args.max_depth,
                        learning_rate=args.learning_rate, 
                        subsample=args.subsample)
    
    model_dir = os.getenv('AIP_MODEL_DIR')
    # GCSFuse conversion
    gs_prefix = 'gs://'
    gcsfuse_prefix = '/gcs/'
    if model_dir.startswith(gs_prefix):
        model_dir = model_dir.replace(gs_prefix, gcsfuse_prefix)
        dirpath = os.path.split(model_dir)[0]
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
    
    gcs_model_path = os.path.join(model_dir, 'model.bst')
    model.save_model(gcs_model_path)
    logging.info(f"Saved model artifacts to {gcs_model_path}")
    

if __name__ == "__main__":
    main()
