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

#Drop Duplicates
df = df.drop(["ID", "No_Pation"], axis=1, errors="ignore")
df = df.drop_duplicates()
print(f"Rows after dropping duplicates: {df.shape[0]}")

# Continuous numeric features
numeric_cols = ['Urea', 'Cr', 'HbA1c', 'Chol',
                'TG', 'HDL', 'LDL', 'VLDL', 'BMI']

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
# MANUAL AGE GROUPING (AGE_GROUP: 0=Young, 1=30-44, 2=45-59, 3=60+)

print("\n" + "=" * 70)
print("AGE GROUPING (Manual Bins)")
print("=" * 70)

age_bins = [0, 30, 45, 60, np.inf]
age_labels = [0, 1, 2, 3]  # numeric codes for the model

df['AGE_GROUP'] = pd.cut(
    df['AGE'],
    bins=age_bins,
    labels=age_labels,
    right=False,          # [(0,30), (30,45), (45,60), (60,∞)]
    include_lowest=True
)

print("AGE_GROUP value counts:")
print(df['AGE_GROUP'].value_counts(dropna=False))

# ============================================================
# BUILD CONTINUOUS MATRIX (ONLY LAB VALUES, NO AGE)

X_numeric = df[numeric_cols].copy()
gender_col = df['Gender'].copy()
age_group_col = df['AGE_GROUP'].cat.codes.copy()  #.cat.codes turns it into integers

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

X_final = X_disc_numeric.copy()
X_final["AGE_GROUP"] = age_group_col.values
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
    min_samples_split=10,       #Prevents the tree from creating branches based on tiny, unreliable subsets.
    min_samples_leaf=5          #A leaf node must contain at least 5 samples.Ensures each final class prediction is based on enough data.
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

print("\n" + "=" * 70)
print("PLOTTING THE DECISION TREE")
print("=" * 70)

# ============================================================
# ANALYSIS: AGE & GENDER VS DIABETES
print("\n" + "=" * 70)
print("AGE & GENDER DIABETES ANALYSIS")
print("=" * 70)

# --------------------------
# AGE GROUP ANALYSIS (<40 vs ≥40) USING ENCODED y (0,1,2)
# --------------------------
age_threshold = 40
young_mask = df['AGE'] < age_threshold
old_mask   = df['AGE'] >= age_threshold

young_y = y[young_mask]
old_y   = y[old_mask]

young_pct = young_y.value_counts(normalize=True) * 100
old_pct   = old_y.value_counts(normalize=True) * 100

print(f"\nDiabetes distribution among YOUNG people (< {age_threshold}):")
for cls, pct in young_pct.items():
    label = {0: "Non-Diabetic (0)", 1: "Pre-Diabetic (1)", 2: "Diabetic (2)"}[cls]
    print(f"  {label}: {pct:.2f}%")

print(f"\nDiabetes distribution among OLDER people (≥ {age_threshold}):")
for cls, pct in old_pct.items():
    label = {0: "Non-Diabetic (0)", 1: "Pre-Diabetic (1)", 2: "Diabetic (2)"}[cls]
    print(f"  {label}: {pct:.2f}%")

# % of younger people who are diabetic (class 2)
if 2 in young_pct.index:
    print(f"\nPercentage of YOUNGER people diagnosed as Diabetic (Class 2): {young_pct[2]:.2f}%")
else:
    print("\nNo younger people fall in Diabetic class (Class 2) in the dataset.")

# --------------------------
# GENDER ANALYSIS (Male vs Female) USING ENCODED y
# --------------------------

gender_pct = pd.crosstab(df['Gender'], y, normalize='index') * 100
print("\nDiabetes percentage by GENDER (rows = Gender, cols = Class 0/1/2):")
print(gender_pct.round(2))

# Male = 1, Female = 2, class 2 = diabetic
male_pct    = gender_pct.loc[1, 2] if (1 in gender_pct.index and 2 in gender_pct.columns) else None
female_pct  = gender_pct.loc[2, 2] if (2 in gender_pct.index and 2 in gender_pct.columns) else None

if male_pct is not None and female_pct is not None:
    print(f"\nPercentage of MALES who are diabetic (Class 2):   {male_pct:.2f}%")
    print(f"Percentage of FEMALES who are diabetic (Class 2): {female_pct:.2f}%")

    if female_pct > male_pct:
        print("\nAccording to the dataset, WOMEN are more prone to diabetes.")
    elif male_pct > female_pct:
        print("\nAccording to the dataset, MEN are more prone to diabetes.")
    else:
        print("\nBoth genders show equal diabetes percentage.")
else:
    print("\nGender analysis could not be completed due to missing data.")


plt.figure(figsize=(30, 20))  
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