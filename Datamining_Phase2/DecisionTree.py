import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv("DatasetofDiabetes.csv")

# Drop duplicates
df = df.drop_duplicates()
print(f"Rows after dropping duplicates: {df.shape[0]}")

# Continuous numeric features
numeric_cols = ['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol',
                'TG', 'HDL', 'LDL', 'VLDL', 'BMI']

feature_cols = numeric_cols + ['Gender']

# ------------------------------------------------------------
#OUTLIER HANDLING (IQR + median)

print("\n" + "=" * 70)
print("OUTLIER HANDLING (IQR + Median Replacement)")
print("=" * 70)

for col in numeric_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    median_val = df[col].median()
    mask = (df[col] < lower) | (df[col] > upper)
    outlier_count = mask.sum()
    df.loc[mask, col] = median_val
    print(f"{col}: {outlier_count} outliers replaced with median {median_val:.3f}")

# ============================================================
#CLASS PREPROCESSING (N,P,Y → 0,1,2)

print("\n" + "=" * 70)
print("TARGET COLUMN (CLASS) PREPROCESSING")
print("=" * 70)

df['CLASS'] = df['CLASS'].astype(str).str.strip().str.upper()
label_map = {'N': 0, 'P': 1, 'Y': 2}
y = df['CLASS'].map(label_map).astype(int)

print("Class counts:")
print(y.value_counts())

# ============================================================
#GENDER PREPROCESSING (M:1, F:2)

print("\n" + "=" * 70)
print("GENDER PREPROCESSING")
print("=" * 70)

df['Gender'] = df['Gender'].astype(str).str.strip().str.upper()
df['Gender'] = df['Gender'].map({'M': 1, 'F': 2})

print(df['Gender'].value_counts(dropna=False))

# ============================================================
#BUILD CONTINUOUS MATRIX

X_numeric = df[numeric_cols].copy()
gender_col = df['Gender'].copy()

print(f"\nFinal dataset size: {len(df)} samples")

# ============================================================
#DISCRETIZATION using KBinsDiscretizer

print("\n" + "=" * 70)
print("DISCRETIZING CONTINUOUS FEATURES USING KBinsDiscretizer (3 bins)")
print("=" * 70)

kbins = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')

X_disc_numeric = pd.DataFrame(
    kbins.fit_transform(X_numeric),
    columns=numeric_cols
)

print("\nSample after KBins discretization:")
print(X_disc_numeric.head())

# Add gender (not discretized)
X_final = X_disc_numeric.copy()
X_final["Gender"] = gender_col.values

# ============================================================
#TRAIN/TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

print("\nData sizes:")
print("Training:", len(X_train))
print("Testing :", len(X_test))

# ============================================================
#DECISION TREE CLASSIFIER

print("\n" + "=" * 70)
print("TRAINING DecisionTreeClassifier (entropy)")
print("=" * 70)

dt = DecisionTreeClassifier(
    criterion='entropy',
    random_state=42,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5
)

dt.fit(X_train, y_train)

y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("\n================ RESULTS ================")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing  Accuracy: {test_acc:.4f}")
print("=========================================")

print("\nClassification Report (Test Set):")
print(classification_report(
    y_test,
    y_test_pred,
    target_names=["Non-Diabetic (0)", "Predict-Diabetic (1)", "Diabetic (2)"],
    digits=3
))

cm = confusion_matrix(y_test, y_test_pred)
cm_df = pd.DataFrame(cm,
                     index=["Actual 0", "Actual 1", "Actual 2"],
                     columns=["Pred 0", "Pred 1", "Pred 2"])
print("\nConfusion Matrix:")
print(cm_df)
# ============================================================
#VISUALIZE THE DECISION TREE

print("\n" + "=" * 70)
print("PLOTTING THE DECISION TREE")
print("=" * 70)

plt.figure(figsize=(30, 20))   # much wider and taller

tree.plot_tree(
    dt,
    feature_names=X_final.columns.tolist(),
    class_names=["Non-Diabetic (N)", "Pre-Diabetic (P)", "Diabetic (Y)"],
    filled=True,
    rounded=True,
    fontsize=11,
    impurity=True,
    proportion=True,
    node_ids=False,
    precision=2,
    max_depth=None
)

plt.title("Decision Tree – Diabetes Classification (Discretized Features)", 
          fontsize=20, pad=30)

# This is the magic line: rotates the long feature names so they don't overlap
plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig("diabetes_tree_large_readable.png", dpi=300, bbox_inches='tight')
plt.show()