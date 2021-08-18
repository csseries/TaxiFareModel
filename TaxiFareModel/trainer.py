# imports
from TaxiFareModel.data import get_data, clean_data, holdout
from TaxiFareModel.pipeline import PreprocPipeline
from TaxiFareModel.utils import compute_rmse

from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient
from ml_flow_test import *
import joblib

MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[DE][MUC][csseries] linear + random forest 2.0"
STORAGE_LOCATION = 'models/simpletaxifare/model.joblib'

class Trainer():
    
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        pipe = PreprocPipeline()
        self.pipeline = pipe.create_pipeline('pickup_datetime')

    def run(self):
        """set and train the pipeline"""
        _df = get_data()
        df = clean_data(_df)

        self.X_train, self.X_test, self.y_train, self.y_test = holdout(df)
        self.pipeline.fit(self.X_train, self.y_train)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(self.X_test)
        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_param('param_name', 0)
        self.mlflow_log_metric('rmse', rmse)
        
        return rmse
    
    def upload_model_to_gcp():

        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(STORAGE_LOCATION)
        blob.upload_from_filename('model.joblib')
    
    def save_model(reg):
        """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
        HINTS : use joblib library and google-cloud-storage"""

        # saving the trained model to disk is mandatory to then beeing able to upload it to storage
        # Implement here
        joblib.dump(reg, 'model.joblib')
        print("saved model.joblib locally")

        # Implement here
        upload_model_to_gcp()
        print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")

    def save_pipeline(self):
        joblib.dump(self.pipeline, 'pipeline.joblib')
    
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(EXPERIMENT_NAME)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)
        


if __name__ == "__main__":
    _df = get_data()
    df = clean_data(_df)
    X = df
    y = df.fare_amount
    X_train, X_test, y_train, y_test = holdout(df)
    trainer = Trainer(X,y)
    trainer.set_pipeline()
    trainer.run()
    experiment_id = trainer.mlflow_experiment_id
    print(f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")
    trainer.evaluate(X_test, y_test)
    #print('TODO')
