import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,roc_auc_score

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

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
scaler=StandardScaler() #doing scaling for logistic regression so that coefficients are interpretable
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

lr=LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled,y_train)

lr_prob=lr.predict_proba(X_test_scaled)[:,1]# whether dev is at burnout risk
lr_pred=(lr_prob>=0.5).astype(int)

print("Logistic Regression Classification Report:")
print(classification_report(y_test,lr_pred))
print("Logistic Regression ROC AUC Score:",roc_auc_score(y_test,lr_prob))

rf=RandomForestClassifier(n_estimators=300,max_depth=10,random_state=42)
rf.fit(X_train,y_train)
rf_prob=rf.predict_proba(X_test)[:,1]

print("Random Forest Classification Report:")
print("ROC-AUC:", roc_auc_score(y_test, rf_prob))

feature_importance = (
    pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": rf.feature_importances_
    })
    .sort_values(by="importance", ascending=False)
)
print("\nFeature Importance (Random Forest):")
print(feature_importance)