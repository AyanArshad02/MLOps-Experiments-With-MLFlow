import mlflow

print("Printing Tracking URI scheme below before setting it")
print("MLflow Tracking URI:", mlflow.get_tracking_uri())
print("|---------------------------------|")

print("Setting the Tracking URI to the local MLflow server")
mlflow.set_tracking_uri("http://localhost:5000")
print("MLflow new Tracking URI:", mlflow.get_tracking_uri())

"""
We need to set_tracking_uri to the local MLflow server because sometimes the default tracking URI
is not set to the local MLflow server and it throws an error when we try to log the metrics, parameters,
and we need to set it to the local MLflow server to log the
metrics, parameters, and artifacts to the local MLflow server.
"""