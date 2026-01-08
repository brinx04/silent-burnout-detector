import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df=pd.read_csv("data/processed/labeled_features.csv")

FEATURE_COLS = [
    "avg_weekly_commits",
    "commit_irregularity",
    "late_night_commit_cent",
    "messages_per_day",
    "avg_response_hours",
    "silent_days",
    "issues_reopen_rate",
    "avg_pr_merge_days"
]
X = df[FEATURE_COLS]
y=df["burnout_risk"]

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
lr=LogisticRegression(max_iter=1000)
lr.fit(X_scaled,y)

coef_df=pd.DataFrame({
    "feature": FEATURE_COLS,
    "coefficient": lr.coef_[0]
})

feature_means=X.mean()
def explain_dev(dev_id, top_k=3):
    row = df[df["dev_id"] == dev_id].iloc[0]
    X_row = row[FEATURE_COLS].values.reshape(1, -1)
    X_row_scaled = scaler.transform(X_row)

    risk_prob = lr.predict_proba(X_row_scaled)[0, 1]

    contributions = coef_df.copy()
    contributions["value"] = row[FEATURE_COLS].values
    contributions["baseline"] = feature_means.values
    contributions["impact"] = (
        contributions["coefficient"]
        * (contributions["value"] - contributions["baseline"])
    )

    top_factors = (
        contributions
        .sort_values(by="impact", ascending=False)
        .head(top_k)
    )

    print(f"\n Developer: {dev_id}")
    print(f"Burnout Risk Probability: {risk_prob:.2f}\n")
    print("Key Contributing Factors:")

    for _, r in top_factors.iterrows():
        direction = "↑" if r["value"] > r["baseline"] else "↓"
        print(f"• {r['feature']} {direction}")

if __name__ == "__main__":
    explain_dev("D14")
