from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
""" Feature selection """
df = pd.read_json('data0910.json' )
df['profit_ratio'] = (df['profit_ratio'].multiply(100)).round(2).astype(str) + '%'


# """ prepare labels """
# long_df = df.loc[(df['is_short'] == False)]
# short_df =  df.loc[(df['is_short'] == True)]

"""  Long experiment   """

features = []
for i in range(1,21):
    features.append(f'rsi_-{i}')
    features.append(f'mfi_-{i}')
    features.append(f'kc_-{i}')
    features.append(f'ema21_-{i}')
    df[f'kc_-{i}'] = df['kc_middleband'].shift(i)
    df[f'ema21_-{i}'] = df['ema21'].shift(i)
    df[f'rsi_-{i}'] = df['rsi'].shift(i)
    df[f'mfi_-{i}'] = df['mfi'].shift(i)
""" prepare labels """
long_df = df.loc[(df['is_short'] == False)]
short_df =  df.loc[(df['is_short'] == True)]


X = df[features]
y = df['label']
X_pipeline = make_pipeline(StandardScaler())
y_pipeline = make_pipeline(OrdinalEncoder())
X = X_pipeline.fit_transform(X)
y = y_pipeline.fit_transform(y.values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8675309)


def run_exps(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """ Run experiments """
    dfs = []
    models = [
          ('LogReg', LogisticRegression()), 
          ('RF', RandomForestClassifier()),
          ('KNN', KNeighborsClassifier()),
          ('SVM', SVC()), 
          ('GNB', GaussianNB()),
          ('XGB', XGBClassifier())
        ]
    results = []
    names = []
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
    # target_names = ['malignant', 'benign']
    reports = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=3, shuffle=True, random_state=90210)
        cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
        clf = model.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # print(name)
        print(name, classification_report(y_test, y_pred))
        reports.append(classification_report(y_test, y_pred))

        results.append(cv_results)
        names.append(name)
        this_df = pd.DataFrame(cv_results)
        this_df['model'] = name
        dfs.append(this_df)
    final = pd.concat(dfs, ignore_index=True)
    return final, reports
result, reports = run_exps(X_train, y_train, X_test, y_test)




