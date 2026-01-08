import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)

NUM_DEVS = 5000
DAYS = 90
developers = [f"D{str(i).zfill(2)}" for i in range(1, NUM_DEVS + 1)]

rows = []
start_date = datetime.today() - timedelta(days=DAYS)

for dev in developers:
    burnout_prone = random.random() < 0.35  # 35% at-risk developers

    for day in range(DAYS):
        date = start_date + timedelta(days=day)

        # baseline commits
        commits = np.random.poisson(2)

        # burnout behavior
        if burnout_prone and day > 45:
            commits += np.random.poisson(2)
            commit_hour = np.random.choice(
                range(21, 24), p=[0.4, 0.4, 0.2]
            )
        else:
            commit_hour = np.random.choice(range(9, 20))

        rows.append([dev,date.date(),commit_hour,commits])

github_df = pd.DataFrame(
    rows,
    columns=["dev_id", "date", "commit_hour", "commits"]
)

github_df.to_csv("data/raw/github_activity.csv", index=False)
print("github_activity.csv generated")

rows = []

for dev in developers:
    burnout_prone = random.random() < 0.35

    for week in range(1, 14):
        issues_opened = np.random.randint(3, 10)
        issues_closed = np.random.randint(2, issues_opened + 1)

        if burnout_prone and week > 7:
            issues_reopened = np.random.randint(2, 5)
            pr_merge_delay = np.random.randint(4, 10)
        else:
            issues_reopened = np.random.randint(0, 2)
            pr_merge_delay = np.random.randint(1, 4)

        rows.append([dev,week,issues_opened,issues_closed,issues_reopened,pr_merge_delay])

issues_df = pd.DataFrame(rows,columns=["dev_id","week","issues_opened","issues_closed","issues_reopened","avg_pr_merge_days"])

issues_df.to_csv("data/raw/issues_activity.csv", index=False)
print("issues_activity.csv generated")

rows = []

for dev in developers:
    burnout_prone = random.random() < 0.35

    for day in range(DAYS):
        date = start_date + timedelta(days=day)

        if burnout_prone and day > 50:
            messages = np.random.randint(0, 3)
            response_time = np.random.uniform(6, 12)
        else:
            messages = np.random.randint(4, 15)
            response_time = np.random.uniform(1, 4)

        rows.append([dev,date.date(),messages,round(response_time, 2)])

comm_df = pd.DataFrame(rows,columns=["dev_id","date","messages_sent","avg_response_hours"])

comm_df.to_csv("data/raw/communication_activity.csv", index=False)
print("communication_activity.csv generated")
