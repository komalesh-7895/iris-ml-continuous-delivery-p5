import pandas as pd
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("Iris.csv")

if "Id" in df.columns:
    df = df.drop(columns=["Id"])

X = df.drop(columns=["Species"])
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

joblib.dump(model, "iris_model.pkl")

metrics = {
    "accuracy": accuracy,
    "training_records": len(X_train),
    "testing_records": len(X_test)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Iris model trained successfully")
print("Accuracy:", accuracy)
