import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

# loads the diabetes dataset
db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# run description (just metadata)
desc = "the simplest possible example"

# connects to the Mlflow tracking server that you started above
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlflow_tracking_examples")

# executes the run
with mlflow.start_run(run_name="no_artifacts_logged", description=desc) as run:
  rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
  rf.fit(X_train, y_train)


