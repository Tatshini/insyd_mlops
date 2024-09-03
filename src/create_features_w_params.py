import pandas as pd
import numpy as np
import pickle
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(data_path):
    # Load data
    file_path = Path(data_path).expanduser()
    infile = open(file_path, 'rb')
    data = pickle.load(infile)
    infile.close()
    return data


def reformat(inp):
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
    df = pd.DataFrame(inp)
    return df


def process_data(data, test_split_perc):
    # Encode and impute values for target variable
    train_rf = reformat(data)
    pd_data = pd.DataFrame(train_rf)
    Y = pd_data['price']
    X = pd_data.drop('price', axis=1)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
      X, Y, test_size=test_split_perc, random_state=5)

    x_train_val = pd.get_dummies(x_train_val, columns=["loc_string", "loc", "type", "subtype", "selltype"])
    x_test = pd.get_dummies(x_test, columns=["loc_string", "loc", "type", "subtype", "selltype"])
    all_columns = set(x_train_val.columns)
    # Add any missing columns to the test dataset with values set to 0
    missing_columns = all_columns - set(x_test.columns)
    for col in missing_columns:
        x_test[col] = 0
    # Reorder columns to match the order in the training dataset
    x_test = x_test[x_train_val.columns]

    x_train_val["bath"] = x_train_val["bath"].fillna(1)
    x_train_val["bed"] = x_train_val["bed"].fillna(1)
    x_train_val["area"] = x_train_val["area"].fillna(int(x_train_val["area"].mean()))
    x_test["bath"] = x_test["bath"].fillna(1)
    x_test["bed"] = x_test["bed"].fillna(1)
    x_test["area"] = x_test["area"].fillna(int(x_train_val["area"].mean()))

    numeric_columns = x_train_val.select_dtypes(exclude=['object']).columns

    scaler = StandardScaler()
    x_train_numeric_scaled = x_train_val.copy()
    x_train_numeric_scaled[numeric_columns] = scaler.fit_transform(x_train_val[numeric_columns])

    x_train_processed = pd.DataFrame(np.hstack((x_train_numeric_scaled[numeric_columns], x_train_val.select_dtypes(include=['object']))))

    preprocessor = ColumnTransformer(
        transformers=[
            ('tfidf_title', TfidfVectorizer(), 'title'),
            ('tfidf_desc', TfidfVectorizer(), 'desc')
        ],
        remainder='passthrough'
    )

    columns = x_train_val.select_dtypes(exclude=['object']).columns.to_list() + x_train_val.select_dtypes(include=['object']).columns.to_list()

    X_train = pd.DataFrame(x_train_processed.to_numpy(), columns=columns)

    numeric_columns = x_test.select_dtypes(exclude=['object']).columns
    x_test_numeric_scaled = x_test.copy()
    x_test_numeric_scaled[numeric_columns] = \
        scaler.transform(x_test[numeric_columns])

    x_test_processed = pd.DataFrame(np.hstack((x_test_numeric_scaled[numeric_columns], x_test.select_dtypes(include=['object']))))
    X_test = pd.DataFrame(x_test_processed.to_numpy(), columns=columns)

    # clf = Pipeline(
    #     steps=[("preprocessor", preprocessor)]
    # )

    # # Create new train and test data using the pipeline
    # clf.fit(X_train, y_train_val)
    # train_new = clf.transform(X_train)
    # test_new = clf.transform(X_test)

    # # Transform to dataframe and save as a csv
    # train_new = pd.DataFrame.sparse.from_spmatrix(train_new)
    # test_new = pd.DataFrame.sparse.from_spmatrix(test_new)
    # train_new['y'] = y_train_val
    # test_new['y'] = y_test
    # return train_new, test_new, clf
    X_train['price'] = y_train_val
    X_test['price'] = y_test
    return X_train, X_test, preprocessor


def save_data(train_new, test_new, train_name, test_name, clf, clf_name):
    train_new.to_csv(train_name)
    test_new.to_csv(test_name)
    # Save pipeline
    with open(clf_name, 'wb') as f:
        pickle.dump(clf, f)


if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["features"]
    data_name = params["data_path"]
    test_split_perc = params["test_split_perc"]
    col_names = ['price', 'title', 'loc_string', 'loc', 'features', 'type', 'subtype', 'selltype', 'desc']
    data = load_data(data_name)
    train_new, test_new, clf = process_data(data, test_split_perc)
    save_data(train_new, test_new, 'data/processed_train_data.csv', 'data/processed_test_data.csv', clf, 'data/pipeline.pkl')
