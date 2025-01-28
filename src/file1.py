import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the wine dataset
data = load_wine()
X = data.data
y = data.target

# Split the data into training and test sets. (0.75, 0.25) split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# The random forest model
max_depth = 10
n_estimators = 10



# code for MLFlow
with mlflow.start_run():
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    print(f"MSE: {mse}")
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix: {confusion}")

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # mlflow.sklearn.log_model(rf, "model")
    # mlflow.log_artifact("confusion_matrix.png")

    # plt.figure(figsize=(10,7))
    # sns.heatmap(confusion, annot=True)
    # plt.savefig("confusion_matrix.png")
    # plt.close()


