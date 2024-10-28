import argparse
import os
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from time import time

def preprocess_data(df):
    """Preprocess the Titanic dataset for model training."""
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df.drop(columns=["Name", "Ticket", "Cabin"], inplace=True)
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    return df

def train_model(dtrain, params, num_round):
    """Train the XGBoost model with distributed support."""
    start_time = time()
    model = xgb.train(params, dtrain, num_boost_round=num_round)
    print(f"Training time: {time() - start_time:.2f} seconds")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--num_round", type=int, default=100)

    args = parser.parse_args()

    # Detect if running on SageMaker or locally
    input_data_path = os.path.join(os.environ.get("SM_CHANNEL_TRAIN", "."), "train.csv")
    output_data_path = os.path.join(os.environ.get("SM_MODEL_DIR", "."), "xgboost-model")

    # Load and preprocess data
    df = pd.read_csv(input_data_path)
    df = preprocess_data(df)
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # Split data for evaluation purposes
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Set up XGBoost parameters for distributed training
    params = {
        "objective": "binary:logistic",
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "eval_metric": "logloss",
    }

    # Train the model and save it
    model = train_model(dtrain, params, args.num_round)
    joblib.dump(model, output_data_path)
    print(f"Model saved to {output_data_path}")
