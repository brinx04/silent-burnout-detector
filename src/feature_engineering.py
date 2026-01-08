import pandas as pd
import numpy as np

#github activity
github=pd.read_csv("data/raw/github_activity.csv")
github["date"]=pd.to_datetime(github["date"]) #converting to datetime format

github["week"]=github["date"].dt.to_period("W")
weekly_commits=(github.groupby(['dev_id','week'])['commits'].sum().reset_index())
github_features = (weekly_commits.groupby("dev_id")["commits"].agg(avg_weekly_commits="mean",commit_irregularity="std").reset_index())
#if standard deviation(std) is high then more burnout risk

late_night_mask=(github["commit_hour"]<6) | (github["commit_hour"]>22)

late_night=(github[late_night_mask].groupby('dev_id')["commits"].sum())

total_commits=github.groupby("dev_id")["commits"].sum()

github_features["late_night_commit_cent"]=(late_night/total_commits*100).fillna(0).values

#communication activity

comm=pd.read_csv("data/raw/communication_activity.csv")
comm["date"]=pd.to_datetime(comm["date"])
comm["week"]=comm["date"].dt.to_period("W")

weekly_comm = (comm.groupby(["dev_id", "week"]).agg(messages_per_day=("messages_sent", "mean"),avg_response_hours=("avg_response_hours", "mean"),silent_days=("messages_sent", lambda x: (x == 0).sum())).reset_index())

comm_features=(weekly_comm.groupby("dev_id").mean().reset_index())

#issues feautures

issues=pd.read_csv("data/raw/issues_activity.csv")

issues_features=(issues.groupby("dev_id").agg(issues_reopen_rate=("issues_reopened","mean"),avg_pr_merge_days=("avg_pr_merge_days","mean")).reset_index())

#Merging all features

features=(github_features.merge(comm_features,on="dev_id",how="left").merge(issues_features,on="dev_id",how="left"))

features.to_csv("data/processed/features.csv",index=False)
print("Feature engineering completed")