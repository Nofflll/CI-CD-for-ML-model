import os
import argparse
import pandas as pd
import yaml
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
import mlflow

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(config_path):
    """
    Trains the model based on the provided configuration.
    """
    logging.info("Starting the training process...")

    # Load parameters from YAML
    with open(config_path) as f:
        config = yaml.safe_load(f)

    base_config = config['base']
    data_split_config = config['data_split']
    train_config = config['train']

    # Start MLflow run
    # mlflow.set_tracking_uri("http://127.0.0.1:5000") # We'll use local logging for now
    with mlflow.start_run():
        logging.info("MLflow run started.")
        mlflow.log_params(base_config)
        mlflow.log_params(data_split_config)
        mlflow.log_params(train_config)

        # Load data
        data_path = os.path.join("data", "raw", "winequality-red.csv")
        logging.info(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path, sep=';')

        # Standardize column names (replace spaces with underscores)
        df.columns = [col.replace(' ', '_') for col in df.columns]

        # Create target variable
        df['quality_label'] = (df['quality'] >= 7).astype(int)
        df = df.drop('quality', axis=1)

        X = df.drop('quality_label', axis=1)
        y = df['quality_label']
        
        # Split data
        logging.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=data_split_config['test_size'], 
            random_state=base_config['random_state']
        )
        
        # Train model
        logging.info("Training RandomForestClassifier...")
        model = RandomForestClassifier(
            n_estimators=train_config['n_estimators'],
            max_depth=train_config['max_depth'],
            random_state=base_config['random_state']
        )
        model.fit(X_train, y_train)
        
        # Evaluate model
        logging.info("Evaluating model...")
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        logging.info(f"  Accuracy: {accuracy:.4f}")
        logging.info(f"  Precision: {precision:.4f}")
        logging.info(f"  Recall: {recall:.4f}")
        logging.info(f"  F1 Score: {f1:.4f}")
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics)

        # Save metrics to JSON file
        metrics_path = "metrics.json"
        logging.info(f"Saving metrics to {metrics_path}")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", "model.joblib")
        logging.info(f"Saving model to {model_path}")
        joblib.dump(model, model_path)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        
        logging.info("Training process finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a machine learning model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (params.yaml).")
    
    args = parser.parse_args()
    
    train_model(args.config) 