from zenml import step
from sklear.metrics import classification_report, accuracy_score
import mlflow

@step
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_pred, y_test)
    accuracy = accuracy_score(y_pred, y_test)
    mlflow.log_metrics("accurcy", accuracy)
    mlflow.log_text(report, "classification_report.txt")
