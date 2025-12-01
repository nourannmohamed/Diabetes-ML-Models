import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv('DatasetofDiabetes.csv')

# ------------------------------------------------------------
# 1) DROP DUPLICATES
# ------------------------------------------------------------
df = df.drop_duplicates()
print(f"Rows after dropping duplicates: {df.shape[0]}")

# ------------------------------------------------------------
# 2) DEFINE NUMERIC FEATURES
# ------------------------------------------------------------
numeric_cols = ['Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']

# Gender will be added to feature columns after mapping
feature_cols = numeric_cols + ['Gender']

# ------------------------------------------------------------
# 3) OUTLIER HANDLING USING IQR (ONLY NUMERIC COLUMNS)
# ------------------------------------------------------------
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

    print(f"{col:<6}: {outlier_count:3d} outliers replaced with median = {median:.2f}")

print(f"\nTotal outliers replaced across all numeric columns: {total_outliers}")

# ------------------------------------------------------------
# 4) PREPROCESS CLASS COLUMN (N,P,Y → 0,1,2)
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("TARGET COLUMN (CLASS) PREPROCESSING")
print("=" * 70)

df['CLASS'] = df['CLASS'].astype(str).str.strip().str.upper()

label_map = {'N': 0, 'P': 1, 'Y': 2}
y = df['CLASS'].map(label_map)

print("\nUnique CLASS values after mapping:")
print(y.value_counts(dropna=False))

# ------------------------------------------------------------
# 5) PREPROCESS GENDER COLUMN (M/F/f → numeric)
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("GENDER PREPROCESSING")
print("=" * 70)

df['Gender'] = df['Gender'].astype(str).str.strip().str.upper()
# Now values should be only 'M' or 'F' (and 'f' becomes 'F' if present)

gender_map = {'M': 1, 'F': 0}
df['Gender'] = df['Gender'].map(gender_map)

print("Unique Gender values after mapping (1=M, 0=F):")
print(df['Gender'].value_counts(dropna=False))

# ------------------------------------------------------------
# 6) KEEP ONLY ROWS WITH VALID LABELS & VALID GENDER
# ------------------------------------------------------------
valid_mask = y.notna() & df['Gender'].notna()

X = df.loc[valid_mask, feature_cols]
y = y.loc[valid_mask].astype(int)

print(f"\nFinal dataset size used for modeling: {X.shape[0]} samples")
print("Feature columns used:", feature_cols)

# ------------------------------------------------------------
# 7) TRAIN / TEST SPLIT
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
# 8) GRID SEARCH FOR BEST CART HYPERPARAMETERS (REGULARIZED)
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("GRID SEARCH: TUNING DECISION TREE HYPERPARAMETERS")
print("=" * 70)

base_dt = DecisionTreeClassifier(
    criterion='gini',   # use 'entropy' if you want info gain instead
    random_state=42
)

# Regularized hyperparameter ranges to reduce overfitting
param_grid = {
    "max_depth": [2, 3, 4, 5],
    "min_samples_split": [10, 20, 50],
    "min_samples_leaf": [5, 10, 20]
}

grid_search = GridSearchCV(
    estimator=base_dt,
    param_grid=param_grid,
    cv=6,                # 6-fold cross-validation
    scoring='accuracy',  # metric to optimize
    n_jobs=-1,           # use all CPU cores
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nBest hyperparameters found:")
print(grid_search.best_params_)

print(f"\nBest cross-validation accuracy: {grid_search.best_score_:.4f}")

# Best model after grid search
best_dt = grid_search.best_estimator_

print("\nBest Decision Tree model:")
print(best_dt)

print(f"\nBest Tree Depth: {best_dt.get_depth()}")
print(f"Best Number of Leaves: {best_dt.get_n_leaves()}")

# ------------------------------------------------------------
# 9) EVALUATE BEST MODEL ON TEST SET
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("EVALUATION OF BEST MODEL ON TEST SET")
print("=" * 70)

y_pred = best_dt.predict(X_test)

# Train vs test accuracy to check overfitting
y_train_pred = best_dt.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_pred)

print(f"\nTrain Accuracy: {train_acc:.4f}")
print(f"Test  Accuracy: {test_acc:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=['Non-Diabetic (0)', 'Predict-Diabetic (1)', 'Diabetic (2)'],
    digits=3
))

cm = pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    index=['Actual 0', 'Actual 1', 'Actual 2'],
    columns=['Pred 0', 'Pred 1', 'Pred 2']
)

print("\nConfusion Matrix:")
print(cm)


from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))  # width, height in inches

tree.plot_tree(
    best_dt,
    feature_names=feature_cols,
    class_names=['Non-Diabetic (0)', 'Predict-Diabetic (1)', 'Diabetic (2)'],
    filled=True,           # color nodes by class
    rounded=True,          # rounded corners
    fontsize=8
)

plt.title("Decision Tree (CART) - Best Model")
plt.show()


# ------------------------------------------------------------
# 10) FEATURE IMPORTANCES OF BEST MODEL
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("FEATURE IMPORTANCES (BEST MODEL)")
print("=" * 70)

importances = pd.Series(best_dt.feature_importances_, index=feature_cols)
print(importances.sort_values(ascending=False))
