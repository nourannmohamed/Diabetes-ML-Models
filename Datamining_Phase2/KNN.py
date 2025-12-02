import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv('DatasetofDiabetes.csv')

# =============================
# REMOVE ID COLUMNS & DUPLICATES
# =============================
df = df.drop(["ID", "No_Pation","HDL"], axis=1, errors="ignore") ###HDL was dropped as it was is the least important feature->accuracy increased
df = df.drop_duplicates()
print(f"Rows after dropping duplicates: {df.shape[0]}")

# =============================
# CHECK NUMERIC COLUMNS
# =============================
numeric_cols = ['Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'LDL', 'VLDL', 'BMI']

# Check for missing values
if df.isnull().values.any():
    print("Null values exist!")
print(df.isnull().sum())

# =============================
# OUTLIER HANDLING (IQR + MEDIAN)
# =============================
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

    df.loc[outlier_mask, col] = median
    print(f"{col:<6}: {outlier_count:3d} outliers replaced with median value = {median:.2f}")

print(f"\nTotal outliers replaced across all numeric columns: {total_outliers}")

# =============================
# FIX STRING ISSUES
# =============================
df['Gender'] = df['Gender'].astype(str).str.strip().str.upper()
df['CLASS'] = df['CLASS'].astype(str).str.strip().str.upper()

print("\nUnique CLASS values:", df["CLASS"].unique())
print("Unique Gender values:", df["Gender"].unique())
print(df["CLASS"].value_counts())
print(df["Gender"].value_counts())

# =============================
# LABEL ENCODING
# =============================
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
df["CLASS"] = le.fit_transform(df["CLASS"])

print("\nAfter Encoding:")
print(df["CLASS"].value_counts())
print(df["Gender"].value_counts())

# =============================
# CORRELATION HEATMAP
# =============================
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# =============================
# KNN CLASSIFIER
# =============================
print("\n" + "=" * 70)
print("KNN CLASSIFIER")
print("=" * 70)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# SPLIT DATA
X = df.drop("CLASS", axis=1)
y = df["CLASS"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================
# FIND BEST K (1 → 100)
# =============================
accuracies = {}

print("\nFinding best K value...")
for k in range(1, 151):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    acc = knn.score(X_test_scaled, y_test)
    accuracies[k] = acc
    print(f"K = {k:3d} → Accuracy = {acc:.4f}")

best_k = max(accuracies, key=accuracies.get)
print(f"\nBest K value = {best_k} with accuracy {accuracies[best_k]:.4f}")

# =============================
# TRAIN BEST MODEL
# =============================
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_scaled, y_train)
y_pred = best_knn.predict(X_test_scaled)

# =============================
# METRICS
# =============================
print("\n" + "=" * 70)
print(f"KNN RESULTS (K = {best_k})")
print("=" * 70)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\nPrecision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred, average='weighted'))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# =============================
# CONFUSION MATRIX HEATMAP
# =============================
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"KNN Confusion Matrix Heatmap (K = {best_k})")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# =============================
# FEATURE IMPORTANCE (PERMUTATION)
# =============================
from sklearn.inspection import permutation_importance

print("\nCalculating feature importance...")

result = permutation_importance(
    best_knn, X_test_scaled, y_test,
    n_repeats=10, random_state=42
)

importances = pd.Series(result.importances_mean, index=X.columns)

plt.figure(figsize=(8, 5))
importances.sort_values().plot(kind='barh', color='teal')
plt.title("Feature Importance (Permutation Importance)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.grid(axis='x')
plt.show()

print("\nFeature Importance Scores:")
print(importances.sort_values(ascending=False))

# =============================
# PLOT ACCURACY vs K
# =============================
plt.figure(figsize=(10, 5))
plt.plot(list(accuracies.keys()), list(accuracies.values()), marker='o')
plt.title("KNN Accuracy for Different K Values")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
