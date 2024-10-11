from metaflow import FlowSpec, step, Flow
import pandas as pd
import numpy as np
from pathlib import Path
import pickle


class PricePredictFlow(FlowSpec):

    @step
    def start(self):
        # Load the latest trained model from the 'PriceTrainFlow'
        self.run = Flow('PriceTrainFlow').latest_run
        self.model = self.run['end'].task.data.best_model
        self.test_config = self.run['end'].task.data.test_config
        print("Model loaded successfully")
        # Load the test data
        self.X_test = self.load_test_data()
        self.next(self.predict)

    @step
    def predict(self):
        # Use the loaded model to make predictions
        self.y_pred = self.model.predict(self.X_test)
        
        # Convert predictions to DataFrame
        self.result_df = pd.DataFrame(self.y_pred, columns=['price'])
        self.result_df.index.name = 'id'
        
        print(f"Predictions made: {self.result_df.head()}")
        self.next(self.save_results)

    @step
    def save_results(self):
        # Save the predictions to a CSV file
        self.result_df.to_csv("solution.csv", index=True)
        print("Results saved to 'solution.csv'")
        self.next(self.end)

    @step
    def end(self):
        print("Prediction flow completed.")

    def load_test_data(self):
        # Replace this with your actual logic to load the test data (X_test)
        # For now, we assume it's loaded similarly as in your original script
        file_path = Path('../data/test_kaggle.pickle').expanduser()
        with open(file_path, 'rb') as infile:
            test = pickle.load(infile)
        test_rf = self.reformat(test)  # Reformat as needed
        x_test = pd.DataFrame(test_rf).drop('id', axis=1)
        # Type hint for Pylint
        x_test: pd.DataFrame  # Explicitly declaring that x_test is a DataFrame
        x_test = x_test[self.test_config["columns_orig"]]
        x_test = pd.get_dummies(x_test, columns=["loc_string", "loc", "type", "subtype", "selltype"])
        all_columns = set(self.test_config["all_colummns"])
        missing_columns = all_columns - set(x_test.columns)
        for col in missing_columns:
            x_test[col] = 0
        self.x_test = x_test[list(all_columns)]

        # Filling missing values
        self.fill_missing_values()
        # Standardizing the data
        self.X_test = self.preprocess_data()

        return self.X_test

    def fill_missing_values(self):
        """Fill missing values in the training and testing datasets."""
        self.x_test["bath"] = self.x_test["bath"].fillna(1)
        self.x_test["bed"] = self.x_test["bed"].fillna(1)
        self.x_test["area"] = self.x_test["area"].fillna(self.test_config['area'])

    def preprocess_data(self):
        # Concatenate numeric and categorical columns side by side
        numeric_columns = self.test_config['numeric_columns']
        x_test_numeric_scaled = self.x_test.copy()
        x_test_numeric_scaled[numeric_columns] = self.test_config['scaler'].transform(self.x_test[numeric_columns])
        self.X_test = pd.concat([x_test_numeric_scaled[numeric_columns], self.x_test.select_dtypes(include=['object'])], axis=1)
        # self.X_test = pd.DataFrame(np.hstack((x_test_numeric_scaled[numeric_columns], self.x_test.select_dtypes(include=['object']))))
        return self.X_test
    
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


if __name__ == '__main__':
    PricePredictFlow()
