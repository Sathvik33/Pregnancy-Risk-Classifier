import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score,recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib


Base_dir=os.path.dirname(__file__)
Data_Path=os.path.join(Base_dir,"Data","pregnancy risk prediction data set.csv")

data = pd.read_csv(Data_Path, encoding='latin1')

print(data)
print(data.shape)
print(data.columns)
print(data.info())
print(data.describe())
print(data.count().isnull())

data.drop(["Patient ID","Name"], axis=1, inplace=True)
# print(data)

print(data["Outcome"].value_counts())



data['Outcome'] = data['Outcome'].str.strip().str.lower()


label_mapping = {
    'low risk': 0,
    'mid risk': 1,
    'high risk': 2
}
data['Outcome'] = data['Outcome'].map(label_mapping)

# Verify the result
print(data['Outcome'].value_counts())

x=data.drop(columns="Outcome")
y=data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}
rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1_weighted',
    verbose=1,
    n_jobs=-1
)

# Fit to training data
grid_search.fit(X_train, y_train)


print("Best Parameters:", grid_search.best_params_)
print("Best F1 Score:", grid_search.best_score_)
best_rf = grid_search.best_estimator_

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Low Risk', 'Mid Risk', 'High Risk']))


importances = best_rf.feature_importances_
features = x.columns

plt.figure(figsize=(8, 6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance (Pregnancy Risk)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


model_path = os.path.join("C:\\C_py\\Project\\Pregnancy risk\\Model", "pregnancy_risk_model.pkl")
joblib.dump(best_rf, model_path)