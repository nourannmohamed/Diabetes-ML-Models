import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# 1. LOAD DATA & DROP DUPLICATES
# ============================================================
print("\n=== DATA LOADING ===")
df = pd.read_csv("DatasetofDiabetes.csv")

df = df.drop(["ID", "No_Pation"], axis=1, errors="ignore")
df = df.drop_duplicates()
print(f"Rows after removing duplicates: {df.shape[0]}")

# ============================================================
# 2. NUMERIC COLUMNS & OUTLIER HANDLING
# ============================================================
numeric_cols = ['Urea', 'Cr', 'HbA1c', 'Chol',
                'TG', 'HDL', 'LDL', 'VLDL', 'BMI']

print("\n=== OUTLIER HANDLING (IQR + Median) ===")
total_outliers = 0

for col in numeric_cols:
    if col not in df.columns:
        print(f"- Skipping missing column: {col}")
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
    print(f"- {col}: {outlier_count} outliers replaced")

print(f"Total outliers replaced: {total_outliers}")

# ============================================================
# 3. GENDER PREPROCESSING (SIMPLE MAPPING)
# ============================================================
print("\n=== GENDER PREPROCESSING ===")

df['Gender'] = df['Gender'].astype(str).str.strip().str.upper()

gender_map = {
    'F': 0,
    'M': 1
}
df['Gender_num'] = df['Gender'].map(gender_map)

print("Gender encoded as: F → 0, M → 1")

# ============================================================
# 4. TARGET COLUMN (CLASS) PREPROCESSING
# ============================================================
print("\n=== CLASS PREPROCESSING ===")

df['CLASS'] = df['CLASS'].astype(str).str.strip().str.upper()

label_map = {'N': 0, 'P': 1, 'Y': 2}
y = df['CLASS'].map(label_map).astype(int)

print("Class counts (after encoding N→0, P→1, Y→2):")
print(y.value_counts())

# ============================================================
# 5. FEATURE MATRIX
# ============================================================
feature_cols = numeric_cols + ['Gender_num']
X = df[feature_cols]

print("\n=== FINAL DATASET USED FOR MODEL ===")
print(f"Samples: {X.shape[0]}")
print(f"Features: {feature_cols}")

# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================
print("\n=== TRAIN / TEST SPLIT ===")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")

# ============================================================
# 7. TRAIN GAUSSIAN NAIVE BAYES
# ============================================================
print("\n=== TRAINING GAUSSIAN NAIVE BAYES ===")

gnb = GaussianNB()
gnb.fit(X_train, y_train)

# ============================================================
# 8. EVALUATION
# ============================================================
print("\n=== MODEL EVALUATION ===")

y_pred = gnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

target_names = ['Non-Diabetic (0)', 'Predict-Diabetic (1)', 'Diabetic (2)']

print("\nClassification report:")
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

print("Confusion matrix:")
print(cm_df)
