from RagBasedStockAnalyser.common.logging_config import setup_logging
from RagBasedStockAnalyser.equity.pipeline.XGBoostEarningModel import XGBoostModel
from RagBasedStockAnalyser.equity.storeData.S3Store import S3Store
import pandas as pd
from sklearn.metrics import r2_score
logger = setup_logging(logger_name=__name__)
def RunModelAndEvaluate(ticker:list, targetMetrics:list,s3:S3Store= S3Store(bucket_name="earnings"))->dict:
    """It trains and evalaute the model against provided metrics"""
    model = XGBoostModel(ticker=ticker)
    predictions = {}
    for target_metric in targetMetrics:
        (X_train, y_train), (X_test, y_test) = preprocess_data(target_metric=target_metric,ticker=ticker,train_cuttoff="CY2023Q4I",s3=s3)
        model.train((X_train, y_train))
        y_pred=model.predict(X_test)
        score= r2_score(y_true=y_test,y_pred=y_pred)

        predictions[target_metric] =score
        logger.info(f"Predictions for {target_metric}: prediction {y_pred} ,Actual :{y_test},score:{score}")
    return predictions
def load_data(ticker=None,s3:S3Store=None)->pd.DataFrame:
        dataFilePath=f"processed_earnings_{'_'.join(ticker)}.csv"
        local_path=f"temp_{dataFilePath}"
        downloaded=s3.download_File(object_name=dataFilePath, file_path=local_path)
        if not downloaded:
            raise FileNotFoundError(f"Could not download data file {dataFilePath} from S3.")
        data = pd.read_csv(local_path)
        return data
def preprocess_data(target_metric,ticker,s3:S3Store=None,train_cuttoff:str="CY2023Q4I")->tuple:
        data=load_data(ticker=ticker,s3=s3)
        df_sorted = data.sort_values(["cik", "frame"])
        feature_cols = [col for col in data.columns if any(suffix in col for suffix in ["_t-1", "_growth", "_accel"])
]
        train_df = df_sorted[df_sorted["frame"] <= train_cuttoff]
        test_df = df_sorted[df_sorted["frame"] > train_cuttoff]
        logger.info(f"Training data shape: {train_df.shape}, Testing data shape: {test_df.shape}")  
        X_train = train_df[feature_cols]
        y_train = train_df[target_metric]
        X_test = test_df[feature_cols]
        y_test = test_df[target_metric]

        return  (X_train, y_train), (X_test, y_test)

