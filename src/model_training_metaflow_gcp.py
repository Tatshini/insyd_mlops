from metaflow import FlowSpec, step, Parameter, conda_base, current
import pandas as pd
import pickle
from pathlib import Path
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import cross_val_score
import mlflow
from google.cloud import storage


@conda_base(libraries={'numpy':'1.26.4', 'scikit-learn':'1.5.1', 'pandas':'2.2.2', 'mlflow':'2.15.1'}, python='3.12.4')
class PriceTrainFlow(FlowSpec):
    cv = Parameter('cv', default=5, type=int, required=True)
    random_seed = Parameter('random_state', default=42, type=int, required=True)

    @step
    def start(self):
        """Load and preprocess the training datasets."""
        bucket_name = 'pricelist_data'
        file_name = 'train/train.pickle'

        # Initialize a GCS client
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)

        # Download the blob as bytes and load it with pickle
        train_data_bytes = blob.download_as_bytes()
        train = pickle.loads(train_data_bytes)

        print("Data loaded successfully")
        train_rf = self.reformat(train)
        self.test_config = {}
        
        # Preparing train sets
        train = pd.DataFrame(train_rf)
        self.y_train = train['price']
        self.x_train = train.drop('price', axis=1)
        self.columns_orig = self.x_train.columns
        self.test_config["columns_orig"] = self.x_train.columns
        # Handling missing columns
        self.x_train = pd.get_dummies(self.x_train, columns=["loc_string", "loc", "type", "subtype", "selltype"])
        # self.columns_with_dummies = self.x_train.columns
        # Filling missing values
        self.fill_missing_values()
        self.test_config["all_colummns"] = set(self.x_train.columns)
        # Standardizing the data
        self.preprocess_data()

        print("Data preprocessed successfully")
        self.next(self.train_gb_model)

    def reformat(self, inp):
        """Reformat the data as needed (you can adapt this function as per your needs)."""
        for i in range(len(inp)):
            try:
                if type(inp[i]['price']) is str:
                    t = ''
                    for a in inp[i]['price']:
                        if a in '0123456789':
                            t += a
                    inp[i]['price'] = int(t)
            except KeyError:
                pass
            try:
                for a in inp[i]['features']:
                    s = a.split(' ')
                    if s[1] == 'm2':
                        inp[i]['area'] = int(s[0])
                    elif s[1] == 'hab.':
                        inp[i]['bed'] = int(s[0])
                    elif s[1] == 'ba\u00f1o':
                        inp[i]['bath'] = int(s[0])
                del inp[i]['features']
            except KeyError:
                pass
        data = pd.DataFrame(inp)
        return data

    def fill_missing_values(self):
        """Fill missing values in the training datasets."""
        self.x_train["bath"] = self.x_train["bath"].fillna(1)
        self.x_train["bed"] = self.x_train["bed"].fillna(1)
        self.x_train["area"] = self.x_train["area"].fillna(int(self.x_train["area"].mean()))
        self.test_config["area"] = int(self.x_train["area"].mean())

    def preprocess_data(self):
        """Standardize numeric columns and process categorical/text features."""
        numeric_columns = self.x_train.select_dtypes(exclude=['object']).columns
        self.test_config["numeric_columns"] = numeric_columns
        scaler = StandardScaler()
        x_train_numeric_scaled = self.x_train.copy()
        x_train_numeric_scaled[numeric_columns] = scaler.fit_transform(self.x_train[numeric_columns])
        self.scaler = scaler
        self.test_config["scaler"] = self.scaler

        preprocessor = ColumnTransformer(
            transformers=[
                ('tfidf_title', TfidfVectorizer(), 'title'),
                ('tfidf_desc', TfidfVectorizer(), 'desc')
            ],
            remainder='passthrough'
        )
        # Select numeric and categorical columns
        numerical_columns = x_train_numeric_scaled[numeric_columns]  # Already scaled numeric columns
        categorical_columns = self.x_train.select_dtypes(include=['object'])  # Categorical columns

        # Concatenate numeric and categorical columns side by side
        self.X_train = pd.concat([numerical_columns, categorical_columns], axis=1)
        self.columns = self.X_train.columns.to_list()

    @step
    def train_gb_model(self):
        """Train GradientBoostingRegressor with different hyperparameters and log the results to MLflow."""
        
        self.best_r2 = 0
        self.best_model = None
        r2_scorer = make_scorer(r2_score)

        mlflow.set_tracking_uri('https://mlflow-539716308541.us-west2.run.app')
        mlflow.set_experiment('price-prediction-metaflow-gcp-experiment')
        param_grid = {
            'n_estimators': [300, 400],
            'learning_rate': [0.08, 0.09, 0.12],
            'max_depth': [3, 5],
            'min_samples_split': [5, 10],
            'subsample': [0.6],
            'random_state': [42]
        }
        for est in param_grid["n_estimators"]:
            for lr in param_grid["learning_rate"]:
                for md in param_grid["max_depth"]:
                    for ss in param_grid["min_samples_split"]:
                        for sbs in param_grid["subsample"]:
                            with mlflow.start_run():
                                mlflow.set_tags({"Model": "GradientBoosting", "Train Data": "training-set"})
                                mlflow.log_params({'n_estimators': est, 'learning_rate': lr, 'max_depth': md, 'min_samples_split': ss, 'subsample': sbs, 'cv': self.cv, 'random_state': self.random_seed})

                                gb_model = make_pipeline(
                                    ColumnTransformer(
                                        transformers=[
                                            ('tfidf_title', TfidfVectorizer(), 'title'),
                                            ('tfidf_desc', TfidfVectorizer(), 'desc')
                                        ],
                                        remainder='passthrough'
                                    ),
                                    GradientBoostingRegressor(n_estimators=est, learning_rate=lr, max_depth=md,
                                                              min_samples_split=ss, random_state=self.random_seed, subsample=sbs)
                                )
                                cv_scores = cross_val_score(gb_model, self.X_train, self.y_train, cv=self.cv, scoring=r2_scorer)
                                mean_r2 = cv_scores.mean()
                                
                                mlflow.log_metric("Mean R2 Score", mean_r2)
                                if mean_r2 > self.best_r2:
                                    self.best_r2 = mean_r2
                                    self.best_model = gb_model
        self.best_model.fit(self.X_train, self.y_train)
        mlflow.sklearn.log_model(self.best_model, artifact_path = 'metaflow_train', registered_model_name="metaflow-price-model-gcp")
        with open('test_config.pkl', 'wb') as f:
            pickle.dump(self.test_config, f)
        mlflow.log_artifact("test_config.pkl", artifact_path='metaflow_train')
        mlflow.end_run()

        self.next(self.end)

    @step
    def end(self):
        """Print out the best model and its score."""
        print(f"Best R2 Score: {self.best_r2}")
        print('Model:', self.best_model)
        print("Best model trained successfully.")

if __name__ == '__main__':
    PriceTrainFlow()