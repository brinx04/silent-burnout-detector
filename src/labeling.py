import pandas as pd

df = pd.read_csv("data/processed/features.csv")

risk_flags = pd.DataFrame({
    "late_night": (df["late_night_commit_cent"] > 8).astype(int),
    "irregular_work": (
        df["commit_irregularity"] >
        df["commit_irregularity"].quantile(0.75)
    ).astype(int),
    "low_communication": (df["messages_per_day"] < 6).astype(int),
    "silent_days": (df["silent_days"] >= 1).astype(int),
    "slow_response": (df["avg_response_hours"] > 5).astype(int)
})

df["burnout_score"] = risk_flags.sum(axis=1)

df["burnout_risk"] = (df["burnout_score"] >= 2).astype(int)

df.to_csv("data/processed/labeled_features.csv", index=False)


print(" Labeling complete")
