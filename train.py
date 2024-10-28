import argparse  # Module for parsing command-line arguments
import os  # Module for interacting with the operating system
import pandas as pd  # Pandas for data manipulation and analysis
from sklearn.model_selection import train_test_split, cross_val_score  # For splitting data and cross-validation
from sklearn.ensemble import RandomForestClassifier  # The RandomForest model from scikit-learn
from sklearn.metrics import accuracy_score  # Accuracy metric to evaluate model performance
import joblib  # Used for saving the trained model

def model_fn(model_dir):
    """
    Load and return a trained model from the specified directory.

    This function is required by SageMaker for deploying the model. SageMaker will
    call this function during inference to load the model from the specified directory.

    Parameters:
    model_dir (str): The directory where the model is saved.

    Returns:
    object: The loaded model object.
    """
    # Load the model file using joblib and return the loaded model object
    return joblib.load(os.path.join(model_dir, "model.joblib"))

if __name__ == "__main__":
    """
    Main function to run the training process. This function sets up the
    training environment, parses hyperparameters and input data paths,
    and trains a RandomForest model using the provided dataset.

    It also performs cross-validation to evaluate the model's performance
    and saves the trained model to a specified directory.

    SageMaker automatically runs this script when a training job is initiated.
    """

    # Set up argument parser for command-line arguments
    parser = argparse.ArgumentParser()

    # Adding command-line arguments for hyperparameters
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the forest.")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of each tree.")

    # Adding command-line argument for the path to the training data
    parser.add_argument("--train", type=str, default="/opt/ml/input/data/train/train.csv", help="Path to training data")

    # Parse the arguments passed to the script
    args = parser.parse_args()

    # Load the dataset from the specified path
    # SageMaker mounts the dataset from S3 to the `/opt/ml/input/data` directory
    df = pd.read_csv(args.train)

    # Split the dataset into features (X) and target variable (y)
    X = df.drop(columns=["Survived"])  # Features (all columns except the target "Survived")
    y = df["Survived"]  # Target column (label)

    # Initialize the RandomForest model with hyperparameters passed via command-line arguments
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,  # Number of trees in the forest
        max_depth=args.max_depth,  # Maximum depth of each tree
        random_state=42  # Set random seed for reproducibility
    )

    # Perform cross-validation on the dataset to evaluate the model's performance
    # `cv=5` means 5-fold cross-validation (splitting data into 5 parts and validating on each)
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    mean_score = scores.mean()  # Calculate the mean accuracy score from cross-validation
    print(f"Cross-Validation Accuracy: {mean_score}")  # Log the mean accuracy score for SageMaker logs

    # Train the model on the entire dataset after cross-validation
    model.fit(X, y)

    # Save the trained model using joblib
    # SageMaker expects models to be saved in `/opt/ml/model` for deployment
    joblib.dump(model, os.path.join("/opt/ml/model", "model.joblib"))
