import pandas as pd
from pathlib import Path

dataDir = Path(r"C:\Users\alan\Desktop\School\Security Analytics\Project\EMBER 2024\data")

files = [path for path in dataDir.rglob("*.jsonl") if "challenge" not in path.name.lower()]

dfs = []

#load data from jsonl files 
for path in files:

    part = pd.read_json(path, lines=True)

    if "file_type" in part.columns:

        part = part[part["file_type"].astype(str).str.upper() == "PDF"]
    if len(part) > 0:

        dfs.append(part)

if len(dfs) == 0:
    raise ValueError("No data found")

#combine all the data into a single dataframe
df = pd.concat(dfs, ignore_index=True)

#remove duplicates based on sha256
print("Rows before duplicates purge :", len(df))

df = df.drop_duplicates(subset=["sha256"], keep="first").reset_index(drop=True)

print("Rows after duplicates gone", len(df))

df["split"] = "excluded"
#temporal split
df.loc[df["week_id"].between(0, 41), "split"] = "train"
df.loc[df["week_id"].between(42, 51), "split"] = "validation"
df.loc[df["week_id"].between(52, 63), "split"] = "test"

#split info
print("\nSplit ")
for splitName in ["train", "validation", "test", "excluded"]:

    part = df[df["split"] == splitName]
    benign = (part["label"] == 0).sum()
    malicious = (part["label"] == 1).sum()

    print(f"{splitName}: rows={len(part):,}, benign={benign:,}, malicious={malicious:,}")

active = df[df["split"].isin(["train", "validation", "test"])]

shaCrossing = (active.groupby("sha256")["split"].nunique() > 1).sum()

print("\nSHA values crossing :", shaCrossing)
