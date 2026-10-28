import pandas as pd

csvPath = r"C:\Users\alan\Desktop\School\Security Analytics\Project\CIC\Obfuscated-MalMem2022.csv"
#load data and basic info
df = pd.read_csv(csvPath)

print("Rows:", len(df))
print("Columns:", len(df.columns))

# counts and percantages 
counts = df["Class"].value_counts()
percentages = df["Class"].value_counts(normalize=True) * 100

print("\nClass counts")
for label, count in counts.items():

    print(f"{label}: {count:,} ({percentages[label]:.2f}%)")

#missing target values 
print("\nMissing target :", df["Class"].isna().sum())

exactDuplicates = df.duplicated(keep="first").sum()
print("Exact duplicate rows:", exactDuplicates)

#duplicate check excluding category feature 
featureCols = [col for col in df.columns if col != "Category"]
featureDuplicates = df.duplicated(subset=featureCols, keep="first").sum()

print("Duplicates excluding Category features:", featureDuplicates)




missingValues = df.isna().sum().sum()

print("Missing values:", missingValues)

