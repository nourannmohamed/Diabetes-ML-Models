import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv('Datamining_Phase2/DatasetofDiabetes.csv')

print(df.shape[0])

# =============================
# REMOVE ID COLUMNS & DUPLICATES
# =============================
df = df.drop(["ID", "No_Pation","HDL"], axis=1, errors="ignore") ###HDL was dropped as it was is the least important feature->accuracy increased
df = df.drop_duplicates()
print(f"Rows after dropping duplicates: {df.shape[0]}")
rows = df.shape[0]

# =============================
# CHECK NUMERIC COLUMNS
# =============================
numeric_cols = ['Urea',"Cr", 'HbA1c', 'Chol', 'TG', 'LDL', 'VLDL', 'BMI']

# Check for missing values
print("null values in dataset:")
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
# CLEAN STRINGS
# =============================
df['Gender'] = df['Gender'].astype(str).str.strip().str.upper()
df['CLASS']  = df['CLASS'].astype(str).str.strip().str.upper()

print("\nBefore Encoding:")
print(df["CLASS"].value_counts())
print(df["Gender"].value_counts())

# =============================
# ONE-HOT ENCODE GENDER
# =============================
ohe = OneHotEncoder(sparse_output=False, drop='first')
gender_encoded = ohe.fit_transform(df[['Gender']])
# Convert to DataFrame
gender_df = pd.DataFrame(gender_encoded, columns=['Gender_Male'], index=df.index)
# Drop original Gender
df = df.drop('Gender', axis=1)
# Add one-hot column
df = pd.concat([df, gender_df], axis=1)
f=df.head()

# =============================
# LABEL ENCODE CLASS (TARGET)
# =============================
le = LabelEncoder()
df["CLASS"] = le.fit_transform(df["CLASS"])
class_names = le.classes_

print("\nAfter Encoding:")
print("CLASS distribution:")
print(df["CLASS"].value_counts())
print("Gender_Male distribution:")
print(df["Gender_Male"].value_counts())


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
# FIND BEST K
# =============================
upper_k = math.ceil(math.sqrt(rows)*1.5)

accuracies = {}

print("\nFinding best K value...")
for k in range(1, upper_k ):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    acc = knn.score(X_test_scaled, y_test)
    accuracies[k] = acc
    print(f"K = {k:3d} → Accuracy = {acc:.4f}")

best_k = max(accuracies, key=accuracies.get)
print(f"\nBest K value = {best_k} with accuracy {accuracies[best_k]:.4f}")

# =============================
# PLOT ERROR vs K for elbow method
# =============================
errors = {k: 1 - acc for k, acc in accuracies.items()}

plt.figure(figsize=(10, 5))
plt.plot(list(errors.keys()), list(errors.values()), marker='o')
plt.title("KNN Error Rate for Different K Values")
plt.xlabel("K")
plt.ylabel("Error Rate")
plt.xticks(range(0, max(errors.keys()) + 1, 2))
plt.grid(True)
plt.show()


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
# PLOT PER-CLASS PRECISION, RECALL, F1
# =============================
from sklearn.metrics import precision_recall_fscore_support

# Compute per-class scores
precisions, recalls, f1s, supports = precision_recall_fscore_support(y_test, y_pred)

# Create DataFrame for plotting
metrics_df = pd.DataFrame({
    "Class": class_names,    # ['N', 'P', 'Y']
    "Precision": precisions,
    "Recall": recalls,
    "F1-score": f1s,
    "Support": supports
})

print("\nPer-Class Metrics:")
print(metrics_df)

# Plot
plt.figure(figsize=(10, 6))
bar_width = 0.25
x = np.arange(len(class_names))

plt.bar(x - bar_width, precisions, width=bar_width, label='Precision')
plt.bar(x, recalls, width=bar_width, label='Recall')
plt.bar(x + bar_width, f1s, width=bar_width, label='F1-score')

plt.xticks(x, class_names)
plt.xlabel("Class")
plt.ylabel("Score")
plt.title("Per-Class Precision, Recall, and F1-score")
plt.ylim(0, 1)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

#####Check overfitting with Train vs Test accuracies and Cross-Validation#####
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
# =============================
# Train vs Test Accuracies (check overfitting)
# =============================
print("\n" + "=" * 70)
print("Train vs Test Accuracies")
train_acc = best_knn.score(X_train_scaled, y_train)
test_acc = best_knn.score(X_test_scaled, y_test)
print("Train accuracy:", train_acc)
print("Test accuracy : ", test_acc)

# Cross-validation (use pipeline so scaling is included)
print("\nCross Validation (StratifiedKFold) accuracies:")
pipeline = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=best_k))])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
print(scores)
print("Mean Accuracy:", scores.mean())



# =============================
# CONFUSION MATRIX HEATMAP
# =============================
# Get original class names from LabelEncoder
class_names = le.classes_
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
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
# MULTI-CLASS ROC & AUC
# =============================
print("\n" + "=" * 70)
print("MULTI-CLASS ROC & AUC")
print("=" * 70)

# Binarize Y for multi-class ROC
classes = sorted(df["CLASS"].unique())  # e.g., [0,1,2]
y_test_bin = label_binarize(y_test, classes=classes)

# Predict probabilities
y_proba = best_knn.predict_proba(X_test_scaled)

# Per-class ROC curves
fpr = {}
tpr = {}
roc_auc = {}

for i, cls in enumerate(classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# ⭐ ADD THIS BLOCK — micro-average ROC
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])




# =============================
# PLOT ROC CURVE
# =============================
plt.figure(figsize=(8, 6))

# Plot each class
for i, cls in enumerate(classes):
    plt.plot(fpr[i], tpr[i],
             label=f"Class {cls} ROC (AUC = {roc_auc[i]:.3f})")

# Plot micro-average
plt.plot(fpr["micro"], tpr["micro"],
         linestyle="--", linewidth=2,
         label=f"Micro-average ROC (AUC = {roc_auc['micro']:.3f})")

# Random guessing line
plt.plot([0, 1], [0, 1], "k--")

plt.title("Multi-Class ROC Curve (KNN)")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend()
plt.grid(True)
plt.show()

# Print scores
print("\nAUC Scores:")
for i, cls in enumerate(classes):
    print(f"Class {cls} AUC: {roc_auc[i]:.3f}")

print(f"Micro-average AUC: {roc_auc['micro']:.3f}")
