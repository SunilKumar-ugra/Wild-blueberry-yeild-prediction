from wildblueberry.entity import ModelEvaluationConfig
from wildblueberry.utils import save_json
import os
import pandas as pd
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix,
                             precision_score,recall_score,f1_score,roc_auc_score,mean_squared_error,mean_absolute_error,r2_score)

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    
    
    def regression_eval_metrics(self,actual, pred):
        mse=mean_squared_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2, mse


    def log_into_mlflow(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        
        model = joblib.load(self.config.model_path)


        train_x = train_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]


        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment("Wild Blueberry Yeild Prediction2")

        with mlflow.start_run(run_name='ElasticNet'):

            y_train_pred=model.predict(train_x)
            y_test_pred=model.predict(test_x)

             #training   performance
            (trn_rmse, trn_mae, trn_r2,trn_mse) = self.regression_eval_metrics(train_y,y_train_pred)
        
            #testing   performance
            (tst_rmse, tst_mae, tst_r2,tst_mse) = self.regression_eval_metrics(test_y,y_test_pred)
            
            # Saving metrics as local
            scores = {"rmse": trn_rmse, "mae": trn_mae, "r2": trn_r2}
            # scores = {"Accuracy":tst_acc,"F1 Score":tst_f1,"Precission":tst_precission,"Recall":tst_recall,}
            save_json(path=Path(self.config.metric_file_name), data=scores)
            
            #log params
            mlflow.log_params(self.config.all_params)
            #train performance log
            mlflow.log_metric("Train Root Mean Square Error", trn_rmse)
            mlflow.log_metric("Train Mean Absolute Error", trn_mae)
            mlflow.log_metric("Train R-Square", trn_r2)
            mlflow.log_metric("Train Mean Square Error", trn_mse)
            
            # test performance log
            mlflow.log_metric("Test Root Mean Square Error", tst_rmse)
            mlflow.log_metric("Test Mean Absolute Error", tst_mae)
            mlflow.log_metric("Test R-Square", tst_r2)
            mlflow.log_metric("Test Mean Square Error", tst_mse)
            

                
            # Model log
            # mlflow.sklearn.log_model(model, str(model_name)+"_model")


            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticNet")
            else:
                mlflow.sklearn.log_model(model, "model")

    
