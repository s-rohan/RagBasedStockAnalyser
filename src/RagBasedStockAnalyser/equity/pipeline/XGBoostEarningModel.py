from RagBasedStockAnalyser.equity.storeData.S3Store import S3Store
from pandas import DataFrame as df
import pandas as pd
import xgboost as xgb
import pickle
import os
from RagBasedStockAnalyser.common.logging_config import setup_logging

logger = setup_logging(logger_name=__name__)    
class XGBoostModel:
    def __init__(self,ticker:list,**params):
        self.s3Model=S3Store(bucket_name="model")  
        self.ticker=ticker
        self.learning_rate=params.get("learning_rate",0.1)
        self.n_estimaters=params.get("n_estimators",100)
        self.max_depth=params.get("max_depth",6)
        self.file_model_path=f"xgboost_model_{'_'.join(ticker)}.model" 
        self.model = xgb.XGBRegressor(n_estimators= self.n_estimaters, max_depth=self.max_depth, random_state=42,learning_rate=self.learning_rate)

        
    
    def train(self,train_data:tuple):
        self.model.fit(train_data[0],train_data[1])   
    def predict(self,test_data):
        model_Pred = self.model.predict(test_data)
        
        return model_Pred
    def save_model_to_s3(self,model_path):
        pickle.dump(self.model, open(model_path, "wb"))
        self.s3Model.upload_file(file_path=model_path, object_name=self.file_model_path)
    def load_model_from_s3(self,model_path):
        self.s3Model.download_File(object_name=self.file_model_path, file_path=model_path)
        if model_path is None or not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")
        self.model = pickle.load(open(model_path, "rb"))


