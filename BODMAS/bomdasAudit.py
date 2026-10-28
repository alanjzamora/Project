import numpy as np
import pandas as pd

#paths 
npzPath = r"C:\Users\alan\Desktop\School\Security Analytics\Project\BODMAS\bodmas.npz"
metaPath = r"C:\Users\alan\Desktop\School\Security Analytics\Project\BODMAS\bodmas_metadata.csv"

#load data and print 
data = np.load(npzPath)

X = data["X"]
y = data["y"]
meta = pd.read_csv(metaPath)

print("\nRows:", X.shape[0])

print("Feature columns:", X.shape[1])

print("Metadata columns:", meta.shape[1])


values, counts = np.unique(y, return_counts=True)

for value, count in zip(values, counts):

    labelName = "Benign" if value == 0 else "Malware"
    percent = count / len(y) * 100

    print(f"{labelName}: {count:,} ({percent:.2f}%)")
#missing values 
missingTarget = pd.Series(y).isna().sum()
print("\nMissing target values:", missingTarget)

if np.issubdtype(X.dtype, np.floating):
    print("Missing features :", np.isnan(X).sum())
    print("Infinite features :", np.isinf(X).sum())

    print("columns  missing values:", np.isnan(X).any(axis=0).sum())
else:
    print("Missing feature:", pd.DataFrame(X).isna().sum().sum())

combined = pd.DataFrame(np.column_stack((X, y)))

rowHashes = pd.util.hash_pandas_object(combined, index=False)
exactDuplicates = rowHashes.duplicated().sum()

print("\nExact duplicates:", exactDuplicates)

#near duplicates using a 20000 row sample 
rng = np.random.default_rng(4371)

sampleIndexes = rng.choice(len(X), size=min(20000, len(X)), replace=False)

xSample = X[sampleIndexes]
ySample = y[sampleIndexes]

# find rows that match with 3 decimal places
roundedSample = pd.DataFrame(np.column_stack((np.round(xSample, 3), ySample)))
roundedHashes = pd.util.hash_pandas_object(roundedSample, index=False)

roundedDuplicates = roundedHashes.duplicated().sum()

exactSample = pd.DataFrame(np.column_stack((xSample, ySample)))

exactSampleHashes = pd.util.hash_pandas_object(exactSample, index=False)

exactSampleDuplicates = exactSampleHashes.duplicated().sum()
nearDuplicates = max(0, roundedDuplicates - exactSampleDuplicates)


print("Rounded duplicate rows:", roundedDuplicates)
print("Exact duplicate rows  sample:", exactSampleDuplicates)

print("Near-duplicate-only rows  sample:", nearDuplicates)

repeatedShaRows = meta["sha"].duplicated().sum()
repeatedShaValues = meta["sha"].value_counts().gt(1).sum()

familyCounts = meta["family"].dropna().value_counts()

# find unique families and observations 
print("\nMissing SHA values:", meta["sha"].isna().sum())
print("Repeated SHA rows:", repeatedShaRows)

print("SHA values count:", repeatedShaValues)

print("Unique families:", familyCounts.size)

print("Families more than one sample:", (familyCounts > 1).sum())

timestamps = pd.to_datetime(meta["timestamp"], errors="coerce", utc=True)


print("Missing timestamps:", timestamps.isna().sum())

print("Earliest :", timestamps.min())
print("Latest :", timestamps.max())
