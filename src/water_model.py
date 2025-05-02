import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dvclive import Live
import yaml


n_estimators = yaml.safe_load(open("dvclive\params.yaml", "r"))["n_estimators"]

data = pd.read_csv("data/water_potability.csv")
train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)


def fill_missing_with_median(df_original):
    df = df_original.copy()
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)
    return df


# Fill missing values with median
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)


X_train = train_processed_data.iloc[:, 0:-1].values
y_train = train_processed_data.iloc[:, -1].values

# n_estimators = 500

clf = RandomForestClassifier(n_estimators=n_estimators)
clf.fit(X_train, y_train)

# save
pickle.dump(clf, open("model.pkl", "wb"))

X_test = test_processed_data.iloc[:, 0:-1].values
y_test = test_processed_data.iloc[:, -1].values


model = pickle.load(open("model.pkl", "rb"))
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)


with Live(save_dvc_exp=True) as live:
    live.log_metric("acc", acc)
    live.log_metric("precision", precision)
    live.log_metric("recall", recall)
    live.log_metric("f1-score", f1_score)

    live.log_param("n_estimators", n_estimators)
