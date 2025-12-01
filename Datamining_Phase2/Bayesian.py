import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv('DatasetofDiabetes.csv')


df = df.drop_duplicates()
print(f"Rows after dropping duplicates: {df.shape[0]}")

# Numeric columns to check (continuous features)
numeric_cols = ['Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']

# ----------------------------
# Replace Outliers with Median (IQR Method)
# ----------------------------
print("\n" + "=" * 70)
print("OUTLIER HANDLING (IQR + Median Replacement)")
print("=" * 70)

total_outliers = 0

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    median = df[col].median()

    outlier_mask = (df[col] < lower) | (df[col] > upper)
    outlier_count = outlier_mask.sum()
    total_outliers += outlier_count

    # Replace outliers with median
    df.loc[outlier_mask, col] = median

    print(f"{col:<6}: {outlier_count:3d} outliers replaced with median value = {median:.2f}")

print(f"\nTotal outliers replaced across all numeric columns: {total_outliers}")

# ----------------------------
# Handle categorical feature: Gender
# ----------------------------
print("\n" + "=" * 70)
print("CATEGORICAL FEATURE (GENDER) PREPROCESSING")
print("=" * 70)

# Clean Gender column (name might be 'Gender' or 'GENDER' depending on dataset)
df['Gender'] = df['Gender'].astype(str).str.strip()

# Clean display: before mapping
print("\n--- Gender distribution BEFORE mapping ---")
gender_counts = df['Gender'].value_counts(dropna=False)
total_rows = len(df)
for g, c in gender_counts.items():
    pct = (c / total_rows) * 100
    print(f"{str(g):>6} : {c:4d} samples ({pct:5.2f}%)")


gender_map = {
    'F': 0, 'Female': 0, 'female': 0, 'f': 0,
    'M': 1, 'Male': 1, 'male': 1, 'm': 1
}
df['Gender_num'] = df['Gender'].map(gender_map)

# If any unmapped values exist, show them and fill with mode
if df['Gender_num'].isna().any():
    print("\nWARNING: Some Gender values could not be mapped:")
    print(df.loc[df['Gender_num'].isna(), 'Gender'].value_counts(dropna=False))
    mode_gender = df['Gender_num'].mode()[0]
    df['Gender_num'] = df['Gender_num'].fillna(mode_gender)
    print(f"\nUnmapped Gender values filled with mode (Gender_num = {mode_gender}).")

# Clean display: after mapping
print("\n--- Gender distribution AFTER mapping (encoded) ---")
gender_num_counts = df['Gender_num'].value_counts(dropna=False).sort_index()
label_map_print = {0: "Female (0)", 1: "Male (1)"}
for g_val, c in gender_num_counts.items():
    label = label_map_print.get(g_val, f"{g_val}")
    pct = (c / total_rows) * 100
    print(f"{label:<10} : {c:4d} samples ({pct:5.2f}%)")

# ----------------------------
# Prepare data for Gaussian Naive Bayes
# ----------------------------
print("\n" + "=" * 70)
print("TARGET COLUMN (CLASS) PREPROCESSING")
print("=" * 70)

# Clean CLASS column (strip spaces)
df['CLASS'] = df['CLASS'].astype(str).str.strip()

print("\nUnique CLASS values after strip:")
print(df['CLASS'].value_counts(dropna=False))

# Map CLASS:
# N  -> 0  (Non-diabetic)
# P  -> 1  (Predict-diabetic)
# Y  -> 2  (Diabetic)
label_map = {'N': 0, 'P': 1, 'Y': 2}
y = df['CLASS'].map(label_map)

# Check for unmapped / NaN labels
if y.isna().any():
    print("\nWARNING: Some CLASS values could not be mapped. Showing them:")
    print(df.loc[y.isna(), 'CLASS'].value_counts(dropna=False))

# Keep only rows with valid labels
valid_mask = y.notna()
y = y.loc[valid_mask].astype(int)

# FEATURES = numeric features + encoded Gender
feature_cols = numeric_cols + ['Gender_num']
X = df.loc[valid_mask, feature_cols]

print(f"\nFinal dataset size used for modeling: {X.shape[0]} samples")
print("Feature columns used:", feature_cols)

# ----------------------------
# Train-test split
# ----------------------------
print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y     # keep class proportions same in train and test
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing  samples: {X_test.shape[0]}")

# ----------------------------
# Train Gaussian Naive Bayes model
# ----------------------------
print("\n" + "=" * 70)
print("TRAINING GAUSSIAN NAIVE BAYES MODEL")
print("=" * 70)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

# ----------------------------
# Evaluate model
# ----------------------------
print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

y_pred = gnb.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")

# Classification Report
target_names = ['Non-Diabetic (0)', 'Predict-Diabetic (1)', 'Diabetic (2)']

print("\nClassification Report:")
print("-" * 70)
print(classification_report(
    y_test,
    y_pred,
    target_names=target_names,
    digits=3
))

# Confusion Matrix as labeled DataFrame
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
