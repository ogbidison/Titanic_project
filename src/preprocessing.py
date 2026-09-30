import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

class DataPreprocessing:
    #def __init__(self, df):
        #self.df = df
        #self.drop_col = drop_col

    def remove_cols(self, df, drop_col):
        #df_drop = df.copy()
        return df.drop(drop_col, axis = 1)

    def replace_val(self, df):
        return df.replace("?", np.nan)

    def encode_val(self, df, columns):
        df = pd.get_dummies(df, columns = columns)

        bool_columns = df.select_dtypes('bool').columns

        df[bool_columns] = df[bool_columns].astype('Int64')
        #df = df.astype({col : "int64" for col in df.select_dtypes('bool').columns})
        return df

    def prep_pipelines (self, age_col, fare_col):
        age_pipeline = Pipeline([
                ('inputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
        
        fare_pipeline =Pipeline([
                ('inputer', SimpleImputer(strategy='most_frequent')),
                ('scaler', StandardScaler())
            ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('age_pipe', age_pipeline, age_col),
                ('fare_pipe', fare_pipeline, fare_col)
            ], remainder='passthrough'
            )
        return preprocessor

    def splitting (self, df, target, test_size, random_state):
        X = df.drop(columns = [target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        return X_train, X_test, y_train, y_test

    def fit_preprocess (self, preprocessor, X_train, X_test):
        preprocessor.fit(X_train)
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        return (X_train_transformed, X_test_transformed, preprocessor)

    def parquet_preprocessed (self, X_train_transformed, X_test_transformed, y_train, y_test):
        X_train_tran_df = pd.DataFrame(X_train_transformed)
        X_test_tran_df = pd.DataFrame(X_test_transformed)
        y_train_df = pd.DataFrame(y_train)
        y_test_df = pd.DataFrame(y_test)

        X_train_tran_df.to_parquet("data/processed/X_train_transformed.parquet", index = False)
        X_test_tran_df.to_parquet("data/processed/X_test_transformed.parquet", index = False)
        y_train_df.to_parquet("data/processed/y_train.parquet", index = False)
        y_test_df.to_parquet("data/processed/y_test.parquet", index = False)

    def joblib_preprocessor (self, preprocessor):
        joblib.dump (preprocessor, "data/processed/preprocessor.joblib", compress = 3)