import numpy as np
import pandas as pd
from pathlib import Path

#file paths 
splitPath = r"C:\Users\alan\Desktop\School\Security Analytics\Project\General\split.csv"
bodmasNpz = r"C:\Users\alan\Desktop\School\Security Analytics\Project\BODMAS\bodmas.npz"

cicCsv = r"C:\Users\alan\Desktop\School\Security Analytics\Project\CIC\Obfuscated-MalMem2022.csv"
emberDir = Path(r"C:\Users\alan\Desktop\School\Security Analytics\Project\EMBER 2024\data")

#getting cleaned version of split file
splitDf = pd.read_csv(splitPath)


# BODMAS

data = np.load(bodmasNpz)
X = data["X"]
y = data["y"].astype(int)

bodmasSplit = splitDf[splitDf["dataset"] == "BODMAS"].copy()
bodmasSplit["rowIndex"] = bodmasSplit["row_id"].str.split(":").str[1].astype(int)


#hash using pandas utility function to check for duplicates across splits
bodmasData = pd.DataFrame(np.column_stack((X, y)))
bodmasHashes = pd.util.hash_pandas_object(bodmasData, index=False)




bodmasTrainIds = bodmasSplit.loc[bodmasSplit["split"] == "train", "rowIndex"]
bodmasValidationIds = bodmasSplit.loc[bodmasSplit["split"] == "validation", "rowIndex"]
bodmasTestIds = bodmasSplit.loc[bodmasSplit["split"] == "test", "rowIndex"]

bodmasValidationHashes = set(bodmasHashes[bodmasValidationIds])
bodmasTestHashes = set(bodmasHashes[bodmasTestIds])


bodmasTrainTest = bodmasHashes[bodmasTrainIds].isin(bodmasTestHashes).sum()
bodmasTrainValidation = bodmasHashes[bodmasTrainIds].isin(bodmasValidationHashes).sum()


# CIC check

cic = pd.read_csv(cicCsv)
cic = cic.drop_duplicates().copy()



cicSplit = splitDf[splitDf["dataset"] == "CIC-MalMem-2022"].copy()
cicSplit["rowIndex"] = cicSplit["row_id"].str.split(":").str[1].astype(int)


#hash using all columns to check for duplicates across splits
cicHashes = pd.util.hash_pandas_object(cic, index=False)


cicTrainIds = cicSplit.loc[cicSplit["split"] == "train", "rowIndex"]
cicValidationIds = cicSplit.loc[cicSplit["split"] == "validation", "rowIndex"]
cicTestIds = cicSplit.loc[cicSplit["split"] == "test", "rowIndex"]

cicValidationHashes = set(cicHashes.loc[cicValidationIds])
cicTestHashes = set(cicHashes.loc[cicTestIds])



cicTrainTest = cicHashes.loc[cicTrainIds].isin(cicTestHashes).sum()
cicTrainValidation = cicHashes.loc[cicTrainIds].isin(cicValidationHashes).sum()

# EMBER check 

frames = []
#load files 
for path in emberDir.rglob("*.jsonl"):
    if "challenge" in path.name.lower():
        continue

    part = pd.read_json(path, lines=True)

    if "file_type" in part.columns:
        part = part[part["file_type"].astype(str).str.upper() == "PDF"]

    if len(part) > 0:

        frames.append(part)

ember = pd.concat(frames, ignore_index=True)

#remove duplicates based on sha256
ember = ember.drop_duplicates(subset=["sha256"]).reset_index(drop=True)

emberSplit = splitDf[splitDf["dataset"] == "EMBER2024_PDF"].copy()
emberSplit["rowIndex"] = emberSplit["row_id"].str.split(":").str[1].astype(int)

emberTrainIds = emberSplit.loc[emberSplit["split"] == "train", "rowIndex"]
emberValidationIds = emberSplit.loc[emberSplit["split"] == "validation", "rowIndex"]
emberTestIds = emberSplit.loc[emberSplit["split"] == "test", "rowIndex"]



emberValidationSha = set(ember.loc[emberValidationIds, "sha256"])
emberTestSha = set(ember.loc[emberTestIds, "sha256"])

emberTrainTest = ember.loc[emberTrainIds, "sha256"].isin(emberTestSha).sum()
emberTrainValidation = ember.loc[emberTrainIds, "sha256"].isin(emberValidationSha).sum()

#leakage results 
print("Training rows duplicating test rows")

print("BODMAS:", bodmasTrainTest)
print("CIC:", cicTrainTest)
print("EMBER:", emberTrainTest)

print("\nTraining rows duplicating validation rows")

print("BODMAS:", bodmasTrainValidation)
print("CIC:", cicTrainValidation)
print("EMBER:", emberTrainValidation)


