from .data_ingestion import DataLoader
from .preprocessing import DataPreprocessing
from .train import Inference

from inspect import getfile
#print(">>> DataLoader is coming from:", getfile(DataLoader))

def main():
    #print(">>> DataLoader is coming from:", getfile(DataLoader))
    # Load data
    loader = DataLoader("data/raw/phpMYEkMl (1).csv")
    df = loader.load_csv()
    num_cols = ['age', 'fare', 'body']
    file_path = "data/processed"
    file_name = "processed.csv"
    loader.save_processed(df,file_path, file_name, num_cols)
    print('Raw data loaded')
    print (df.head())

    # Preprocess
    prep = DataPreprocessing()

    # Remove cols
    drop_col = ['name','home.dest', 'ticket', 'cabin', 'body', 'boat']
    df = prep.remove_cols(df, drop_col)

    # Replace Values
    df = prep.replace_val(df)

    # Encode
    columns = ['pclass', 'sex', 'embarked']
    df = prep.encode_val(df, columns)

    # Scaling and Imputing
    age_col = ['age']
    fare_col = ['fare']
    preprocessor = prep.prep_pipelines (age_col, fare_col)

    # Splitting
    target = 'survived'
    test_size = 0.2
    random_state = 42
    X_train, X_test, y_train, y_test = prep.splitting (df, target, test_size, random_state)

    # Fit/Transform
    X_train_transformed, X_test_transformed, preprocessor = prep.fit_preprocess (preprocessor, X_train, X_test)

    # Saving preprocessed in paquet format
    prep.parquet_preprocessed (X_train_transformed, X_test_transformed, y_train, y_test)

    # Save in joblib format
    prep.joblib_preprocessor (preprocessor)

    # Infeence

    #Define parquet path
    X_train_path ="data/processed/X_train_transformed.parquet"
    X_test_path = "data/processed/X_test_transformed.parquet"
    y_train_path = "data/processed/y_train.parquet"
    y_test_path = "data/processed/y_test.parquet"

    #initialize inference engine
    engine = Inference(
        X_train_pq = X_train_path,
        X_test_pq = X_test_path,
        y_train_pq = y_train_path,
        y_test_pq = y_test_path
        )

    y_pred, f1, acc = engine.run_experiment()

    print ("Inference completed")
    print (f"Accuracy: {acc: .4f}")
    print (f"F1score: {f1: .4f}")
    print ("Sample predictions", y_pred[:10])

if __name__ == "__main__":
    main()