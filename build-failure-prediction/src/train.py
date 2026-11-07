"""Steps 2-5: clean data, engineer features, train models, evaluate."""
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)

# Step 2: load + clean
df = pd.read_csv("data/builds.csv")
print("Missing values before:\n", df.isna().sum()[df.isna().sum() > 0])

# Step 3: feature engineering
df["total_churn"] = df.lines_added + df.lines_deleted
df["churn_per_file"] = df.total_churn / df.files_changed
df["tests_per_commit"] = df.tests_run / df.num_commits
df["night_build"] = ((df.build_hour >= 22) | (df.build_hour <= 4)).astype(int)

X = df.drop(columns=["build_id", "build_failed"])
y = df["build_failed"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}
results, fitted = {}, {}
for name, clf in models.items():
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()), ("clf", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results[name] = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
    }
    fitted[name] = (pipe, pred)
    print(name, results[name])

# Step 5: evaluate - save metrics, confusion matrix, feature importance
best = max(results, key=lambda k: results[k]["f1"])
print("Best model:", best)
json.dump({"results": results, "best_model": best},
          open("results/metrics.json", "w"), indent=2)
joblib.dump(fitted[best][0], "results/best_model.joblib")

ConfusionMatrixDisplay(confusion_matrix(y_test, fitted[best][1]),
                       display_labels=["Pass", "Fail"]).plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best}")
plt.savefig("results/confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.close()

rf = fitted["Random Forest"][0].named_steps["clf"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
imp.plot(kind="barh", figsize=(7, 5), color="#2563eb")
plt.title("Feature Importance - Random Forest")
plt.savefig("results/feature_importance.png", dpi=120, bbox_inches="tight")
plt.close()

pd.DataFrame(results).T.plot(kind="bar", figsize=(8, 5))
plt.title("Model Comparison"); plt.ylim(0, 1); plt.xticks(rotation=0)
plt.savefig("results/model_comparison.png", dpi=120, bbox_inches="tight")
print("Done. Outputs saved in results/")
