import pandas as pd
#from .preprocessing import DataPreprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report
import mlflow
import mlflow.sklearn

class Inference:
    def __init__(self, X_train_pq, X_test_pq, y_train_pq, y_test_pq):
        self.X_train_pq = pd.read_parquet(X_train_pq)
        self.X_test_pq = pd.read_parquet(X_test_pq)
        self.y_train_pq = pd.read_parquet(y_train_pq)
        self.y_test_pq = pd.read_parquet(y_test_pq)

    def run_experiment(self):   #X_train_pq, X_test_pq, y_train_pq, y_test_pq, y_pred):
        model = LogisticRegression()
        mlflow.set_experiment("titanic dsaster")

        with mlflow.start_run():
            mlflow.sklearn.autolog()

            #Train
            model.fit (self.X_train_pq, self.y_train_pq)

            #Predict
            y_pred = model.predict(self.X_test_pq)

            #Metrics
            f1 = f1_score(self.y_test_pq, y_pred)
            acc = accuracy_score(self.y_test_pq, y_pred)

            # Manual logging
            #mlflow.log_params()
            mlflow.log_metric("accuracy score", acc)
            mlflow.log_metric("f1 score", f1)

        return y_pred, f1, acc