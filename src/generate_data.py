"""Step 1: Generate a synthetic CI build history dataset (builds.csv)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 3000

df = pd.DataFrame({
    "build_id": np.arange(1, N + 1),
    "num_commits": rng.integers(1, 15, N),
    "files_changed": rng.integers(1, 60, N),
    "lines_added": rng.integers(0, 1500, N),
    "lines_deleted": rng.integers(0, 800, N),
    "tests_run": rng.integers(20, 500, N),
    "test_coverage": rng.uniform(40, 98, N).round(1),
    "author_fail_rate": rng.uniform(0, 0.5, N).round(3),
    "prev_build_failed": rng.integers(0, 2, N),
    "build_hour": rng.integers(0, 24, N),
    "dependency_changed": rng.integers(0, 2, N),
})

# Hidden rule: failure risk grows with change size, dependency changes,
# previous failures and low coverage.
score = (
    0.002 * df.lines_added + 0.002 * df.lines_deleted
    + 0.03 * df.files_changed + 0.05 * df.num_commits
    + 1.2 * df.dependency_changed + 1.0 * df.prev_build_failed
    + 3.0 * df.author_fail_rate - 0.03 * df.test_coverage
    + 0.4 * ((df.build_hour >= 22) | (df.build_hour <= 4))
)
prob = 1 / (1 + np.exp(-(score - score.mean()) * 1.3))
df["build_failed"] = (rng.uniform(0, 1, N) < prob).astype(int)

# Inject some missing values to practice cleaning
for col in ["test_coverage", "author_fail_rate"]:
    df.loc[rng.choice(N, 60, replace=False), col] = np.nan

df.to_csv("data/builds.csv", index=False)
print(f"Saved data/builds.csv  shape={df.shape}")
print(df.build_failed.value_counts(normalize=True).round(3))
