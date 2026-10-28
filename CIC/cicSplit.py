import pandas as pd
from sklearn.model_selection import train_test_split

csvPath = r"C:\Users\alan\Desktop\School\Security Analytics\Project\CIC\Obfuscated-MalMem2022.csv"


df = pd.read_csv(csvPath)


#remove duplicates before split 
duplicateMask = df.duplicated(keep="first")
duplicateCount = duplicateMask.sum()

dfClean = df[~duplicateMask].copy()
dfClean["originalRowIndex"] = dfClean.index
dfClean = dfClean.reset_index(drop=True)

print("Original rows:", len(df))
print("Duplicates removed:", duplicateCount)
print("Rows remaining:", len(dfClean))


# split data using 60/20/20 split 
trainDf, tempDf = train_test_split(dfClean, test_size=0.40, random_state=4371, stratify=dfClean["Class"])
validationDf, testDf = train_test_split(tempDf, test_size=0.50, random_state=4371, stratify=tempDf["Class"])

trainDf = trainDf.copy()
validationDf = validationDf.copy()
testDf = testDf.copy()

trainDf["split"] = "train"
validationDf["split"] = "validation"

testDf["split"] = "test"

#print info for each split
print("\nSplit ")
for splitName, part in [("train", trainDf), ("validation", validationDf), ("test", testDf)]:

    benign = (part["Class"] == "Benign").sum()
    malware = (part["Class"] == "Malware").sum()

    print(f"{splitName}: rows={len(part):,}, benign={benign:,}, malware={malware:,}")



#checking for exact duplicates across splits

combined = pd.concat([trainDf, validationDf, testDf], ignore_index=True)

featureCols = df.columns.tolist()
combined["exactHash"] = pd.util.hash_pandas_object(combined[featureCols], index=False).values

crossingGroups = (combined.groupby("exactHash")["split"].nunique() > 1).sum()

print("\nExact duplicate:", crossingGroups)


