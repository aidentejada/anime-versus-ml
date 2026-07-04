"""
Anime Versus ML - Model Training Pipeline
Module: src/train.py
Description: Ingests the synthetically labeled matchup data and trains an XGBoost
             Classifier to discover non-linear stat interactions and proxy logic
             (e.g., speed blitzing, hax thresholds). Saves the serialized model.
Author: Aiden J. Tejada
Date: 2026-05-22
License: MIT License
"""

__version__ = "0.1.0"
__author__ = "Aiden J. Tejada"
__license__ = "MIT License"

import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import GridSearchCV
import sklearn

df = pd.read_csv("../data/matchups_winners_FIXED.csv") # 2346 rows instead of the mirrored final dataset to prevent leakage.
drop = [
    "winner",
    "a_name",
    "a_form",
    "a_anime",
    "b_name",
    "b_form",
    "b_anime"
]
X = df.drop(drop, axis=1)
y = df["winner"]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)


#mirroring training data ONLY.
a_cols = [c for c in X_train.columns if c.startswith("a_")]
b_cols = [c for c in X_train.columns if c.startswith("b_")]
rename_map = {c: "b_" + c[2:] for c in a_cols} | {c: "a_" + c[2:] for c in b_cols}
X_mirror = X_train.rename(columns=rename_map)[X_train.columns]
y_mirror = 1.0 - y_train
X_train = pd.concat([X_train, X_mirror], ignore_index=True)
y_train = pd.concat([y_train, y_mirror], ignore_index=True)


#params for grid search
params = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
}
#initializing grid search
grid = GridSearchCV(xgb.XGBClassifier(n_estimators=300, early_stopping_rounds=30),
                    params, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

print(f"Best params: {grid.best_params_}")
print(f"Best CV accuracy: {grid.best_score_ * 100:.2f}%")

#training the model on the best parameters.
model = xgb.XGBClassifier(
    n_estimators=1000, # high number because we'll be using stopping rounds.
    learning_rate=grid.best_params_['learning_rate'],
    max_depth=grid.best_params_['max_depth'],
    subsample=grid.best_params_['subsample'],
    colsample_bytree=grid.best_params_['colsample_bytree'],
    min_child_weight=grid.best_params_['min_child_weight'],
    random_state=67,
    early_stopping_rounds=30
)

model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=10 #print out every 10 rounds.
          )


y_pred = model.predict(X_test)

accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

shap_values = shap.TreeExplainer(model).shap_values(X_test)
shap.summary_plot(shap_values, X_test)

model.save_model("../models/xgb_model.json")

