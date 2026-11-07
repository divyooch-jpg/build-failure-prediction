"""Predict failure risk for a single new build."""
import joblib, pandas as pd

model = joblib.load("results/best_model.joblib")
b = {"num_commits": 9, "files_changed": 40, "lines_added": 1200,
     "lines_deleted": 500, "tests_run": 100, "test_coverage": 55.0,
     "author_fail_rate": 0.35, "prev_build_failed": 1, "build_hour": 23,
     "dependency_changed": 1}
b["total_churn"] = b["lines_added"] + b["lines_deleted"]
b["churn_per_file"] = b["total_churn"] / b["files_changed"]
b["tests_per_commit"] = b["tests_run"] / b["num_commits"]
b["night_build"] = int(b["build_hour"] >= 22 or b["build_hour"] <= 4)
X = pd.DataFrame([b])[model.feature_names_in_]
p = model.predict_proba(X)[0][1]
print(f"Failure probability: {p:.2%} -> {'LIKELY FAIL' if p > 0.5 else 'LIKELY PASS'}")
