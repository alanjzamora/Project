import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier


#dataset paths 
bodmasNpz = r"C:\Users\alan\Desktop\School\Security Analytics\Project\BODMAS\bodmas.npz"
bodmasMeta = r"C:\Users\alan\Desktop\School\Security Analytics\Project\BODMAS\bodmas_metadata.csv"

cicCsv = r"C:\Users\alan\Desktop\School\Security Analytics\Project\CIC\Obfuscated-MalMem2022.csv"

emberDir = Path(r"C:\Users\alan\Desktop\School\Security Analytics\Project\EMBER 2024\data")
outputDir = Path(r"C:\Users\alan\Desktop\School\Security Analytics\Project\General")

splitPath = outputDir / "split.csv"
scoresPath = outputDir / "scores.csv" # results paths

#hyperparamaters for XG boost 
modelSettings = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 4371
}

splitRows = []
validationLabels = []
validationScores = []
testResults = {}
trainingTimes = {}

# rows for split file
def addSplitRows(datasetName, rowIds, labels, splitName):
    for rowId, label in zip(rowIds, labels):
        splitRows.append({"row_id": f"{datasetName}:{rowId}", "dataset": datasetName, "split": splitName, "label": int(label)})

#ember feature expansion functions
def expandListColumn(df, columnName, prefix):
    maxLength = df[columnName].apply(lambda value: len(value) if isinstance(value, (list, tuple, np.ndarray)) else 0).max()
    values = np.zeros((len(df), maxLength), dtype=np.float32)

    for row, value in enumerate(df[columnName]):
        if isinstance(value, (list, tuple, np.ndarray)):
            values[row, :len(value)] = np.asarray(value, dtype=np.float32)

    return pd.DataFrame(values, index=df.index, columns=[f"{prefix}{i}" for i in range(maxLength)])

#EMBER dictionary changed to numeric columns
def expandDictColumn(df, columnName, prefix):
    rows = []

    for value in df[columnName]:
        row = {}
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, (int, float, np.integer, np.floating)):
                    row[f"{prefix}{key}"] = float(item)
        rows.append(row)

    return pd.DataFrame(rows, index=df.index).fillna(0)




print("BODMAS")
# load dataset and evaluate BODMAS
data = np.load(bodmasNpz)
xBodmas = data["X"]
yBodmas = data["y"].astype(int)

meta = pd.read_csv(bodmasMeta)

meta["timestampParsed"] = pd.to_datetime(meta["timestamp"], errors="coerce", utc=True)
meta["rowIndex"] = np.arange(len(meta))
meta["split"] = "excluded"

trainMask = meta["timestampParsed"].notna() & (meta["timestampParsed"] >= "2019-08-29") & (meta["timestampParsed"] < "2020-01-01")
validationMask = meta["timestampParsed"].notna() & (meta["timestampParsed"] >= "2020-01-01") & (meta["timestampParsed"] < "2020-05-01")
testMask = meta["timestampParsed"].notna() & (meta["timestampParsed"] >= "2020-05-01") & (meta["timestampParsed"] < "2020-10-01")

meta.loc[trainMask, "split"] = "train"
meta.loc[validationMask, "split"] = "validation"
meta.loc[testMask, "split"] = "test"


#near duplicate check with 20000 sample 
rng = np.random.default_rng(4371)
sampleIndexes = rng.choice(len(xBodmas), size=min(20000, len(xBodmas)), replace=False)

xSample = xBodmas[sampleIndexes]
ySample = yBodmas[sampleIndexes]

sampleInfo = meta.iloc[sampleIndexes].copy().reset_index(drop=True)
sampleInfo["originalRowIndex"] = sampleIndexes

exactSample = pd.DataFrame(np.column_stack((xSample, ySample)))
sampleInfo["exactHash"] = pd.util.hash_pandas_object(exactSample, index=False).values

roundedSample = pd.DataFrame(np.column_stack((np.round(xSample, 3), ySample)))
sampleInfo["nearHash"] = pd.util.hash_pandas_object(roundedSample, index=False).values

nearRemove = []

for _, group in sampleInfo.groupby("nearHash"):
    if len(group) <= 1 or group["exactHash"].nunique() == 1:
        continue

    active = group[group["split"].isin(["train", "validation", "test"])]

    if active["split"].nunique() > 1:
        active = active.sort_values("timestampParsed")
        nearRemove.extend(active.iloc[1:]["originalRowIndex"].tolist())

nearRemove = sorted(set(nearRemove))
meta.loc[meta["rowIndex"].isin(nearRemove), "split"] = "removed"

bodmasTrainIds = meta.loc[meta["split"] == "train", "rowIndex"].to_numpy()
bodmasValidationIds = meta.loc[meta["split"] == "validation", "rowIndex"].to_numpy()

bodmasTestIds = meta.loc[meta["split"] == "test", "rowIndex"].to_numpy()

xTrain = xBodmas[bodmasTrainIds]

yTrain = yBodmas[bodmasTrainIds]
xValidation = xBodmas[bodmasValidationIds]
yValidation = yBodmas[bodmasValidationIds]

xTest = xBodmas[bodmasTestIds]
yTest = yBodmas[bodmasTestIds]


#training using XGBoost
bodmasModel = XGBClassifier(**modelSettings)

start = time.perf_counter()
bodmasModel.fit(xTrain, yTrain)
trainingTimes["BODMAS"] = time.perf_counter() - start

bodmasValidationScores = bodmasModel.predict_proba(xValidation)[:, 1]
bodmasTestScores = bodmasModel.predict_proba(xTest)[:, 1]

validationLabels.extend(yValidation.tolist())
validationScores.extend(bodmasValidationScores.tolist())

testResults["BODMAS"] = {"rowIds": bodmasTestIds, "labels": yTest, "scores": bodmasTestScores}

addSplitRows("BODMAS", bodmasTrainIds, yTrain, "train")

addSplitRows("BODMAS", bodmasValidationIds, yValidation, "validation")
addSplitRows("BODMAS", bodmasTestIds, yTest, "test")

print("Near duplicates removed:", len(nearRemove))

print("Train:", len(yTrain), "Validation:", len(yValidation), "Test:", len(yTest))
print("Training time:", round(trainingTimes["BODMAS"], 2), "seconds")

#CIC-MalMem-2022 eval 
print("\nCIC-MalMem-2022")

cic = pd.read_csv(cicCsv)

cic = cic.drop_duplicates(keep="first").copy()
cic["originalRowIndex"] = cic.index

cic["target"] = cic["Class"].map({"Benign": 0, "Malware": 1}).astype(int)
cic = cic.reset_index(drop=True)

cicFeatureCols = [col for col in cic.columns if col not in ["Category", "Class", "target", "originalRowIndex"]]

#60/20/20 split for CIC-MalMem-2022
cicTrain, cicTemp = train_test_split(cic, test_size=0.40, random_state=4371, stratify=cic["target"])
cicValidation, cicTest = train_test_split(cicTemp, test_size=0.50, random_state=4371, stratify=cicTemp["target"])

xTrain = cicTrain[cicFeatureCols]
yTrain = cicTrain["target"]

xValidation = cicValidation[cicFeatureCols]
yValidation = cicValidation["target"]

xTest = cicTest[cicFeatureCols]

yTest = cicTest["target"]

#train using XGBoost

cicModel = XGBClassifier(**modelSettings)

start = time.perf_counter()

cicModel.fit(xTrain, yTrain)
trainingTimes["CIC-MalMem-2022"] = time.perf_counter() - start

cicValidationScores = cicModel.predict_proba(xValidation)[:, 1]
cicTestScores = cicModel.predict_proba(xTest)[:, 1]

validationLabels.extend(yValidation.tolist())
validationScores.extend(cicValidationScores.tolist())

cicTestIds = cicTest["originalRowIndex"].to_numpy()
testResults["CIC-MalMem-2022"] = {"rowIds": cicTestIds, "labels": yTest.to_numpy(), "scores": cicTestScores}

addSplitRows("CIC-MalMem-2022", cicTrain["originalRowIndex"].to_numpy(), cicTrain["target"].to_numpy(), "train")

addSplitRows("CIC-MalMem-2022", cicValidation["originalRowIndex"].to_numpy(), cicValidation["target"].to_numpy(), "validation")
addSplitRows("CIC-MalMem-2022", cicTestIds, yTest.to_numpy(), "test")

print("Train:", len(yTrain), "Validation:", len(yValidation), "Test:", len(yTest))

print("Training time:", round(trainingTimes["CIC-MalMem-2022"], 2), "seconds")


print("\nEMBER2024 PDF")
#load json files 
files = [path for path in emberDir.rglob("*.jsonl") if "challenge" not in path.name.lower()]
frames = []

for path in files:
    part = pd.read_json(path, lines=True)

    if "file_type" in part.columns:
        part = part[part["file_type"].astype(str).str.upper() == "PDF"]

    if len(part) > 0:
        frames.append(part)

if len(frames) == 0:
    raise ValueError("No EMBER PDF data found")

#remove duplicates based on sha256
ember = pd.concat(frames, ignore_index=True)
ember = ember.drop_duplicates(subset=["sha256"], keep="first").reset_index(drop=True)
ember["originalRowIndex"] = np.arange(len(ember))

emberTrain = ember[ember["week_id"].between(0, 41)].copy()
emberValidation = ember[ember["week_id"].between(42, 51)].copy()
emberTest = ember[ember["week_id"].between(52, 63)].copy()

#set of features for EMBER2024 PDF
featureParts = [

    expandDictColumn(ember, "general", "general"),
    expandListColumn(ember, "histogram", "hist"),
    expandListColumn(ember, "byteentropy", "byteentropy"),
    expandDictColumn(ember, "strings", "strings")
]

emberFeatures = pd.concat(featureParts, axis=1).astype(np.float32).replace([np.inf, -np.inf], 0).fillna(0)

xTrain = emberFeatures.loc[emberTrain.index]
yTrain = emberTrain["label"].astype(int)

xValidation = emberFeatures.loc[emberValidation.index]
yValidation = emberValidation["label"].astype(int)

xTest = emberFeatures.loc[emberTest.index]
yTest = emberTest["label"].astype(int)


#training using XGBoost
emberModel = XGBClassifier(**modelSettings)

start = time.perf_counter()
emberModel.fit(xTrain, yTrain)
trainingTimes["EMBER2024 PDF"] = time.perf_counter() - start

emberValidationScores = emberModel.predict_proba(xValidation)[:, 1]
emberTestScores = emberModel.predict_proba(xTest)[:, 1]

validationLabels.extend(yValidation.tolist())

validationScores.extend(emberValidationScores.tolist())

emberTestIds = emberTest["originalRowIndex"].to_numpy()

testResults["EMBER2024_PDF"] = {"rowIds": emberTestIds, "labels": yTest.to_numpy(), "scores": emberTestScores}

addSplitRows("EMBER2024_PDF", emberTrain["originalRowIndex"].to_numpy(), emberTrain["label"].astype(int).to_numpy(), "train")
addSplitRows("EMBER2024_PDF", emberValidation["originalRowIndex"].to_numpy(), emberValidation["label"].astype(int).to_numpy(), "validation")
addSplitRows("EMBER2024_PDF", emberTestIds, yTest.to_numpy(), "test")

print("Features:", emberFeatures.shape[1])

print("Train:", len(yTrain), "Validation:", len(yValidation), "Test:", len(yTest))

print("Training time:", round(trainingTimes["EMBER2024 PDF"], 2), "seconds")

#get threshold from pooled validation scores 
validationLabels = np.asarray(validationLabels)
validationScores = np.asarray(validationScores)

fpr, tpr, thresholds = roc_curve(validationLabels, validationScores)
valid = (fpr <= 0.05) & np.isfinite(thresholds)

bestRecall = tpr[valid].max()
bestIndexes = np.where(valid & (tpr == bestRecall))[0]
bestIndex = bestIndexes[np.argmin(thresholds[bestIndexes])]

threshold = float(thresholds[bestIndex])

print("\nThreshold:", threshold)

print("Validation FPR:", fpr[bestIndex])
print("Validation recall:", tpr[bestIndex])

#write to checkpoint files 
splitDf = pd.DataFrame(splitRows)
splitDf.to_csv(splitPath, index=False)

scoreRows = []

for testSetName, result in testResults.items():
    for rowId, label, score in zip(result["rowIds"], result["labels"], result["scores"]):
        scoreRows.append({"row_id": f"{testSetName}:{rowId}", "test_set": testSetName, "label": int(label), "score": float(score)})

scoresDf = pd.DataFrame(scoreRows)

scoresDf.to_csv(scoresPath, index=False)

print("\nWrote split.csv:", len(splitDf))

print("Wrote scores.csv:", len(scoresDf))


print("\n times")
for name, seconds in trainingTimes.items():
    print(name + ":", round(seconds, 2), "seconds")

print(f'python check_checkpoint.py --split split.csv --scores scores.csv --threshold {threshold}')
