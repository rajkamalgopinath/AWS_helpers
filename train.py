import argparse
import os
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

def preprocess_data(df):
    """Preprocess the Titanic dataset for model training."""
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df.drop(columns=["Name", "Ticket", "Cabin"], inplace=True)
    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex'])
    df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
    return df

def train_model(X, y, n_estimators=100, max_depth=None):
    """Train the Random Forest model and perform cross-validation."""
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {scores.mean()}")
    model.fit(X, y)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--train", type=str, default="train.csv")

    args = parser.parse_args()
    
    # Load and preprocess the data
    df = pd.read_csv(args.train)
    df = preprocess_data(df)
    
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    
    # Train the model
    model = train_model(X, y, n_estimators=args.n_estimators, max_depth=args.max_depth)

    # Save the model locally
    joblib.dump(model, "model.joblib")
    print("Model saved to model.joblib")
