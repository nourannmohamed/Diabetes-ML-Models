import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ============================================================
# 1) LOAD & BASIC CLEANING
# ============================================================
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv("DatasetofDiabetes.csv")

# Drop duplicates
df = df.drop_duplicates()
print(f"Rows after dropping duplicates: {df.shape[0]}")

# Numeric features (same as you used)
numeric_cols = ['Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']
feature_cols = numeric_cols + ['Gender']

# ------------------------------------------------------------
# OPTIONAL: SIMPLE OUTLIER HANDLING (IQR + median)
# (You can comment this block out if you want it simpler)
# ------------------------------------------------------------
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
# 2) TARGET (CLASS) PREPROCESSING:  N,P,Y  →  0,1,2
# ============================================================
print("\n" + "=" * 70)
print("TARGET COLUMN (CLASS) PREPROCESSING")
print("=" * 70)

df['CLASS'] = df['CLASS'].astype(str).str.strip().str.upper()
label_map = {'N': 0, 'P': 1, 'Y': 2}
y = df['CLASS'].map(label_map)

print("Class counts (before dropping NaN):")
print(y.value_counts(dropna=False))

# ============================================================
# 3) GENDER PREPROCESSING: M,F  →  1,0
# ============================================================
print("\n" + "=" * 70)
print("GENDER PREPROCESSING")
print("=" * 70)

df['Gender'] = df['Gender'].astype(str).str.strip().str.upper()
df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})

print(df['Gender'].value_counts(dropna=False))

# ============================================================
# 4) REMOVE INVALID ROWS
# ============================================================
mask = y.notna() & df['Gender'].notna()
df = df.loc[mask].copy()
y = y.loc[mask].astype(int)

X_numeric = df[numeric_cols].copy()
gender_col = df['Gender'].copy()

print(f"\nFinal dataset size after cleaning: {len(df)} samples")

# ============================================================
# 5) DISCRETIZATION OF NUMERIC FEATURES
#    (Simple dynamic intervals -> Low, Medium, High)
#    This follows the idea from the slides: partition continuous
#    attributes into intervals to get discrete-valued attributes.
# ============================================================
print("\n" + "=" * 70)
print("DISCRETIZING NUMERICAL FEATURES (3 bins: 0=Low,1=Med,2=High)")
print("=" * 70)

def discretize_dataframe(X_num, n_bins=3):
    """
    Discretize each numeric column into n_bins using quantiles.
    Returns:
      - X_disc: DataFrame with integer codes (0..n_bins-1)
    """
    X_disc = pd.DataFrame(index=X_num.index)
    for col in X_num.columns:
        # qcut uses quantiles so each bin has (roughly) equal size
        # duplicates='drop' handles constant columns
        bins = pd.qcut(X_num[col], q=n_bins, duplicates='drop')
        codes, uniques = pd.factorize(bins, sort=True)
        X_disc[col] = codes
        print(f"{col}: unique intervals -> {list(uniques)}")
    return X_disc

X_disc_numeric = discretize_dataframe(X_numeric, n_bins=3)

# Add gender as already discrete {0,1}
X_final = X_disc_numeric.copy()
X_final['Gender'] = gender_col.values

print("\nSample of discretized features:")
print(X_final.head())

# ============================================================
# 6) TRAIN/TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

print("\nData sizes:")
print("Training:", len(X_train))
print("Testing :", len(X_test))

# ============================================================
# 7) ID3 IMPLEMENTATION FOR DISCRETE ATTRIBUTES
# ============================================================

def entropy(y):
    """
    Entropy H(Y) = - sum p_i log2 p_i
    """
    values, counts = np.unique(y, return_counts=True)
    prob = counts / len(y)
    return -np.sum(prob * np.log2(prob + 1e-9))  # add epsilon to avoid log(0)


def information_gain_discrete(X_col, y):
    """
    Information Gain for a discrete attribute A:
    Gain(S, A) = Entropy(S) - sum_v |S_v|/|S| * Entropy(S_v)
    """
    H_parent = entropy(y)
    values, counts = np.unique(X_col, return_counts=True)
    weighted_entropy = 0.0

    for v, cnt in zip(values, counts):
        y_v = y[X_col == v]
        weighted_entropy += (cnt / len(X_col)) * entropy(y_v)

    return H_parent - weighted_entropy


def best_attribute(X, y):
    """
    Find the attribute (column) with maximum information gain.
    """
    best_attr = None
    best_gain = -1

    for col in X.columns:
        gain = information_gain_discrete(X[col], y)
        if gain > best_gain:
            best_gain = gain
            best_attr = col

    return best_attr, best_gain


class Node:
    """
    A simple ID3 tree node:
      - feature: attribute name used for splitting at this node
      - children: dict {attribute_value: child_node}
      - value: if leaf node, the predicted class
    """
    def __init__(self, feature=None, children=None, value=None):
        self.feature = feature
        self.children = children if children is not None else {}
        self.value = value


def majority_class(y):
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]


def id3(X, y, depth=0, max_depth=None):
    """
    ID3 recursive algorithm (for discrete attributes).
    Stopping conditions:
      - all labels are same  -> leaf
      - no attributes left   -> leaf with majority class
      - optional: max_depth reached -> leaf with majority class
    """
    # 1) If all samples have the same class → leaf
    if len(np.unique(y)) == 1:
        return Node(value=y.iloc[0])

    # 2) If no attributes left to split on → leaf with majority class
    if X.shape[1] == 0:
        return Node(value=majority_class(y))

    # 3) If max depth is reached → leaf with majority class
    if (max_depth is not None) and (depth >= max_depth):
        return Node(value=majority_class(y))

    # 4) Choose best attribute (max info gain)
    best_attr, best_gain = best_attribute(X, y)

    # If no gain (or extremely small), make a leaf
    if best_gain <= 1e-6:
        return Node(value=majority_class(y))

    # 5) Create internal node and split data by attribute values
    node = Node(feature=best_attr, children={})
    attr_values = X[best_attr].unique()

    for v in attr_values:
        mask = X[best_attr] == v
        X_subset = X.loc[mask].drop(columns=[best_attr])
        y_subset = y.loc[mask]

        # If subset is empty (shouldn't happen with this mask), make leaf
        if len(y_subset) == 0:
            child = Node(value=majority_class(y))
        else:
            child = id3(X_subset, y_subset, depth=depth + 1, max_depth=max_depth)

        node.children[v] = child

    return node


def predict_one(node, sample):
    """
    Traverse the tree for one sample (a row from X).
    """
    # If leaf node
    if node.value is not None:
        return node.value

    # Internal node: read sample's value for this feature
    attr_value = sample[node.feature]

    # Go to the corresponding child
    child = node.children.get(attr_value, None)

    # If unseen value, fall back to majority at this node (simple handling)
    if child is None:
        # majority vote among children leaves
        leaf_values = []
        for c in node.children.values():
            if c.value is not None:
                leaf_values.append(c.value)
        if len(leaf_values) > 0:
            return max(set(leaf_values), key=leaf_values.count)
        else:
            # fallback safe value
            return 0

    return predict_one(child, sample)


def predict(tree, X):
    return np.array([predict_one(tree, X.iloc[i]) for i in range(len(X))])

# ============================================================
# 8) BUILD TREE & EVALUATE
# ============================================================
print("\n" + "=" * 70)
print("TRAINING ID3 TREE (on discretized features)")
print("=" * 70)

# You can set max_depth=None to allow full growth,
# or limit it, e.g. max_depth=5, to reduce overfitting.
tree = id3(X_train, y_train, max_depth=5)

y_train_pred = predict(tree, X_train)
y_test_pred = predict(tree, X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("\n================ RESULTS ================")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing  Accuracy: {test_acc:.4f}")
print("=========================================")
