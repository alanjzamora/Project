import json
import pandas as pd
from pathlib import Path

dataDir = Path(r"C:\Users\alan\Desktop\School\Security Analytics\Project\EMBER 2024\data")

#find files for the data 
jsonlFiles = list(dataDir.rglob("*.jsonl"))
dfs = []

for path in jsonlFiles:
    if "challenge" in path.name.lower():
        continue

    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            if str(record.get("file_type", "")).upper() == "PDF":

                rows.append(record)

    if rows:

        dfs.append(pd.DataFrame(rows))

if len(dfs) == 0:
    raise ValueError("No data found")


#combine all the data into a single dataframe

df = pd.concat(dfs, ignore_index=True)

#had to remove duplicates because the downloaded file has copies using sha
df = df.drop_duplicates(subset=["sha256"], keep="first").reset_index(drop=True)


print("Columns:", len(df.columns))

counts = df["label"].value_counts(dropna=False)
percentages = df["label"].value_counts(normalize=True, dropna=False) * 100


for label, count in counts.items():

    labelName = "Benign" if label == 0 else "Malicious" if label == 1 else str(label)

    print(f"{labelName}: {count:,} ({percentages[label]:.2f}%)")

#missing target along with sha duplicates check
print("\nMissing target values:", df["label"].isna().sum())

repeatedShaRows = df["sha256"].duplicated().sum()
repeatedShaValues = df["sha256"].value_counts().gt(1).sum()

print("\nMissing SHA values:", df["sha256"].isna().sum())
print("Repeated SHA rows:", repeatedShaRows)
print("Repeated SHA values:", repeatedShaValues)

dates = pd.to_datetime(df["first_submission_date"], unit="s", errors="coerce", utc=True)

#date range
print("\nMissing time:", dates.isna().sum())

print("Earliest:", dates.min())
print("Latest:", dates.max())

#week range
print("\nMin:", df["week_id"].min())
print("Max:", df["week_id"].max())

print("Unique:", df["week_id"].nunique())


#family counts

familyRows = df["family"].dropna()


print("Rows with family:", len(familyRows))
print("Unique families:", familyRows.nunique())

