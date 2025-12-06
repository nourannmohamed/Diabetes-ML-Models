import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv("DatasetofDiabetes.csv")

# ------------------------------------------------------------
# DROP DUPLICATES
# ------------------------------------------------------------
df = df.drop(["ID", "No_Pation"], axis=1, errors="ignore")
df = df.drop_duplicates()
print(f"Rows after dropping duplicates: {df.shape[0]}")

# ------------------------------------------------------------
# NUMERIC COLUMNS
# ------------------------------------------------------------
numeric_cols = ['Urea', 'Cr', 'HbA1c', 'Chol',
                'TG', 'HDL', 'LDL', 'VLDL', 'BMI']

# ------------------------------------------------------------
# OUTLIER HANDLING (IQR + Median Replacement)
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("OUTLIER HANDLING (IQR + Median Replacement)")
print("=" * 70)

total_outliers = 0
for col in numeric_cols:
    if col not in df.columns:
        print(f"Skipping missing numeric column: {col}")
        continue

    df[col] = pd.to_numeric(df[col], errors='coerce')

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    median = df[col].median()

    outlier_mask = (df[col] < lower) | (df[col] > upper)
    outlier_count = int(outlier_mask.sum())
    total_outliers += outlier_count

    df.loc[outlier_mask, col] = median
    print(f"{col:<6}: {outlier_count:3d} outliers replaced with median = {median}")

print(f"\nTotal outliers replaced across numeric columns: {total_outliers}")

# ------------------------------------------------------------
# CATEGORICAL FEATURE (GENDER) PREPROCESSING - SIMPLE MAPPING
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("CATEGORICAL FEATURE (GENDER) PREPROCESSING")
print("=" * 70)

# Clean and standardize gender text
df['Gender'] = df['Gender'].astype(str).str.strip().str.upper()

# Simple mapping to numeric
gender_map = {
    'F': 0, 
    'M': 1
}
df['Gender_num'] = df['Gender'].map(gender_map)

# ------------------------------------------------------------
# TARGET COLUMN (CLASS) PREPROCESSING
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("TARGET COLUMN (CLASS) PREPROCESSING")
print("=" * 70)

# Clean and standardize CLASS values
df['CLASS'] = df['CLASS'].astype(str).str.strip().str.upper()

# Map CLASS to numeric labels
label_map = {'N': 0, 'P': 1, 'Y': 2}
y = df['CLASS'].map(label_map).astype(int)

# Show class distribution
print("Class counts:")
print(y.value_counts())


# ------------------------------------------------------------
# FEATURE MATRIX
# ------------------------------------------------------------
feature_cols = numeric_cols + ['Gender_num']
X = df[feature_cols]


print(f"\nFinal dataset size used for modeling: {X.shape[0]} samples")
print("Feature columns used:", feature_cols)

# ------------------------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing  samples: {X_test.shape[0]}")

# ------------------------------------------------------------
# TRAINING GAUSSIAN NAIVE BAYES MODEL
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("TRAINING GAUSSIAN NAIVE BAYES MODEL")
print("=" * 70)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

# ------------------------------------------------------------
# MODEL EVALUATION
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

y_pred = gnb.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")

target_names = ['Non-Diabetic (0)', 'Predict-Diabetic (1)', 'Diabetic (2)']

print("\nClassification Report:")
print("-" * 70)
print(classification_report(
    y_test,
    y_pred,
    target_names=target_names,
    digits=3
))

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {name}" for name in target_names],
    columns=[f"Pred {name}" for name in target_names]
)

print("\nConfusion Matrix:")
print("-" * 70)
print(cm_df)
print("-" * 70)
