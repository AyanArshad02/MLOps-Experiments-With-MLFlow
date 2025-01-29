# Here we are using the autologging feature of mlflow to log the parameters, metrics and artifacts of the model.
# We can do autolog in both local and remote server.
# Here i have tried autolog in remote server.
# We only need to add mlflow.autolog() in the code to enable autologging.

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

# Got this from Remote -> Experiments -> Using MLFlow Tracking & MLFlow Tracking Remote from dagshub repo
import dagshub
dagshub.init(repo_owner='AyanArshad02', repo_name='MLOps-Experiments-With-MLFlow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/AyanArshad02/MLOps-Experiments-With-MLFlow.mlflow")

# Only above 3 lines of codes are needed for Remote Tracking with MLflow Server i.e. line 16,17,19

mlflow.autolog() # used for autologging
mlflow.set_experiment("wine-classification")

# Load the wine dataset
data = load_wine()
X = data.data
y = data.target

# Split the data into training and test sets. (0.75, 0.25) split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# The random forest model
max_depth = 10
n_estimators = 5



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

    # Plot the confusion matrix
    plt.figure(figsize=(10,7))
    sns.heatmap(confusion, annot=True,fmt='d',cmap='Blues',xticklabels=data.target_names,yticklabels=data.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # We need to log our code file. Autolog does not log the code file.
    mlflow.log_artifact(__file__) # log the source code file (file1.py)

    # set tags
    mlflow.set_tags({"Author": "Ayan", "Framework": "MLFlow"})

    print(mlflow.get_tracking_uri())