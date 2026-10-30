import numpy as np
import pandas as pd
#paths 
npzPath = r"C:\Users\alan\Desktop\School\Security Analytics\Project\BODMAS\bodmas.npz"
metaPath = r"C:\Users\alan\Desktop\School\Security Analytics\Project\BODMAS\bodmas_metadata.csv"

#load data 
data = np.load(npzPath)
X = data["X"]
y = data["y"]
meta = pd.read_csv(metaPath)

meta["timestampParsed"] = pd.to_datetime(meta["timestamp"], errors="coerce", utc=True)

audit = meta.copy()

audit["label"] = y

audit["rowIndex"] = np.arange(len(audit))
audit["split"] = "excluded"

#temporal split based on specific ranges 
trainMask = audit["timestampParsed"].notna() & (audit["timestampParsed"] >= "2019-08-29") & (audit["timestampParsed"] < "2020-01-01")
validationMask = audit["timestampParsed"].notna() & (audit["timestampParsed"] >= "2020-01-01") & (audit["timestampParsed"] < "2020-05-01")

testMask = audit["timestampParsed"].notna() & (audit["timestampParsed"] >= "2020-05-01") & (audit["timestampParsed"] < "2020-10-01")

audit.loc[trainMask, "split"] = "train"
audit.loc[validationMask, "split"] = "validation"

audit.loc[testMask, "split"] = "test"

#split data
print("Split")
for splitName in ["train", "validation", "test", "excluded"]:

    part = audit[audit["split"] == splitName]
    benign = (part["label"] == 0).sum()
    malware = (part["label"] == 1).sum()
    print(f"{splitName}: rows={len(part):,}, benign={benign:,}, malware={malware:,}")

missingTimestampCount = audit["timestampParsed"].isna().sum()
preOverlapCount = (audit["timestampParsed"].notna() & (audit["timestampParsed"] < "2019-08-29")).sum()


print("\nMissinge timestamp:", missingTimestampCount)
print("Before overlap:", preOverlapCount)

print("Total:", (audit["split"] == "excluded").sum())

combined = pd.DataFrame(np.column_stack((X, y)))
audit["exactHash"] = pd.util.hash_pandas_object(combined, index=False).values

exactGroups = audit.groupby("exactHash").filter(lambda group: len(group) > 1)

exactGroupCount = exactGroups["exactHash"].nunique()
exactCrossGroups = 0

exactRowsToRemove = []


#exact duplicates 
for _, group in exactGroups.groupby("exactHash"):

    active = group[group["split"].isin(["train", "validation", "test"])]
    if active["split"].nunique() > 1:

        exactCrossGroups += 1

        active = active.sort_values("timestampParsed")
        exactRowsToRemove.extend(active.iloc[1:]["rowIndex"].tolist())


print("\nGroups:", exactGroupCount)
print("Groups crossing :", exactCrossGroups)
print("Rows to remove:", len(exactRowsToRemove))

#sample audit for near duplicates
rng = np.random.default_rng(4371)
sampleIndexes = rng.choice(len(X), size=min(20000, len(X)), replace=False)

xSample = X[sampleIndexes]
ySample = y[sampleIndexes]

sampleAudit = audit.iloc[sampleIndexes].copy().reset_index(drop=True)
sampleAudit["originalRowIndex"] = sampleIndexes

exactSample = pd.DataFrame(np.column_stack((xSample, ySample)))

sampleAudit["exactHash"] = pd.util.hash_pandas_object(exactSample, index=False).values

# duplicates after 3 places of rounding 
roundedSample = pd.DataFrame(np.column_stack((np.round(xSample, 3), ySample)))
sampleAudit["nearHash"] = pd.util.hash_pandas_object(roundedSample, index=False).values

nearGroups = sampleAudit.groupby("nearHash").filter(lambda group: len(group) > 1)

nearCrossGroups = 0
nearRowsToRemove = []

# keep earliest rows when getting rid of duplicates 
for _, group in nearGroups.groupby("nearHash"):
    if group["exactHash"].nunique() == 1:
        continue

    active = group[group["split"].isin(["train", "validation", "test"])]

    if active["split"].nunique() > 1:
        nearCrossGroups += 1
        active = active.sort_values("timestampParsed")

        nearRowsToRemove.extend(active.iloc[1:]["originalRowIndex"].tolist())

nearRowsToRemove = sorted(set(nearRowsToRemove))


print("Groups crossing :", nearCrossGroups)
print("Rows to remove:", len(nearRowsToRemove))


removedNearRows = audit.loc[nearRowsToRemove]

trainRemoved = (removedNearRows["split"] == "train").sum()
validationRemoved = (removedNearRows["split"] == "validation").sum()
testRemoved = (removedNearRows["split"] == "test").sum()

#removed from splits 
print("\nTrain removed:", trainRemoved)
print("Validation removed:", validationRemoved)
print("Test removed:", testRemoved)

#results of audit 
familyRows = audit[audit["family"].notna() & audit["split"].isin(["train", "validation", "test"])]
familySplitCounts = familyRows.groupby("family")["split"].nunique()


print("\n Families active splits:", len(familySplitCounts))

print("Families crossing splits:", (familySplitCounts > 1).sum())

