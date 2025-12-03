import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

print("=" * 70)
print("EXPLORATORY ANALYSIS")
print("=" * 70)

df = pd.read_csv('DatasetofDiabetes.csv')

# =============================
# REMOVE ID COLUMNS & DUPLICATES
# =============================
df = df.drop(["ID", "No_Pation"], axis=1, errors="ignore")
df = df.drop_duplicates()
print(f"Rows after dropping duplicates: {df.shape[0]}")

# =============================
# CHECK NUMERIC COLUMNS (robust)
# =============================
numeric_cols = ['Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']

# Check for missing values
if df.isnull().values.any():
    print("Warning: Null values exist!")
print(df.isnull().sum())

# =============================
# OUTLIER HANDLING (IQR + MEDIAN) - skip missing cols
# =============================
print("\n" + "=" * 70)
print("OUTLIER HANDLING (IQR + Median Replacement)")
print("=" * 70)

total_outliers = 0
for col in numeric_cols:
    if col not in df.columns:
        print(f"Skipping missing numeric column: {col}")
        continue
    # convert to numeric if possible (coerce errors)
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

    # replace outliers with median (only where value is not NaN)
    df.loc[outlier_mask, col] = median
    print(f"{col:<6}: {outlier_count:3d} outliers replaced with median = {median}")

print(f"\nTotal outliers replaced across numeric columns: {total_outliers}")

# =============================
# FIX STRING ISSUES (USE EXACT COLUMN NAMES YOU HAVE)
# =============================
# Keep column names exactly: AGE, CLASS, Gender
# Normalize values inside Gender/CLASS for reliable comparisons
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].astype(str).str.strip().str.upper()

if 'CLASS' in df.columns:
    df['CLASS'] = df['CLASS'].astype(str).str.strip().str.upper()

# =============================
# Normalize CLASS values — map Y/N/P and common variants
# =============================
if 'CLASS' in df.columns:
    print("\nRaw CLASS unique values:", df['CLASS'].unique())

    df['CLASS'] = df['CLASS'].replace({
        'Y': 'DIABETIC',
        'N': 'NONDIABETIC',
        'P': 'PREDIABETIC',
        'YES': 'DIABETIC',
        'NO': 'NONDIABETIC',
        'YEA': 'DIABETIC',  # just in case
        # add other mappings if needed
    })

    # final cleanup: strip/upper again to be safe
    df['CLASS'] = df['CLASS'].astype(str).str.strip().str.upper()
    print("Cleaned CLASS unique values:", df['CLASS'].unique())
else:
    raise KeyError("CLASS column not found in dataframe. Make sure your CSV contains CLASS column.")

# Safe print of Gender values
if 'Gender' in df.columns:
    print("Cleaned Gender unique values:", df['Gender'].unique())
else:
    print("Gender column not found in dataframe.")

# ------------------------------------------------------
# Basic dataset overview
# ------------------------------------------------------
print("\nDataset shape:", df.shape)
print("\nBasic info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe(include='all'))

# ------------------------------------------------------
# 1. Distribution of diabetes classes
# ------------------------------------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="CLASS", order=sorted(df['CLASS'].unique()))
plt.title("Distribution of Diabetes Classes")
plt.tight_layout()
plt.show()

# ------------------------------------------------------
# 2. Gender distribution (if present)
# ------------------------------------------------------
if 'Gender' in df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x="Gender", order=sorted(df['Gender'].unique()))
    plt.title("Gender Distribution")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------
# 3. Diabetes percentage by gender (robust)
# ------------------------------------------------------
if 'Gender' in df.columns:
    print("\nCalculating diabetes rates by gender...")

    gender_counts = df.groupby("Gender").size()
    diabetic_counts_gender = df[df["CLASS"] == "DIABETIC"].groupby("Gender").size()

    # reindex to include all genders, fill zeros
    diabetic_counts_gender = diabetic_counts_gender.reindex(gender_counts.index, fill_value=0)
    gender_diabetes_percent = (diabetic_counts_gender / gender_counts * 100).fillna(0)

    print("\n% Diabetes by Gender:\n", gender_diabetes_percent)

    plt.figure(figsize=(6,4))
    gender_diabetes_percent.plot(kind='bar', color=["#ff99c8", "#89cff0"][:len(gender_diabetes_percent)])
    plt.title("Percentage of Diabetics by Gender")
    plt.ylabel("Percentage %")
    plt.tight_layout()
    plt.show()
else:
    gender_diabetes_percent = pd.Series(dtype=float)
    print("\nGender column missing — skipping gender-based diabetes percentage.")

# ------------------------------------------------------
# 4. Diabetes percentage by AGE group
# ------------------------------------------------------
if 'AGE' in df.columns:
    # ensure AGE is numeric
    df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')
    df['AGE_Group'] = pd.cut(df['AGE'], bins=[0,30,50,120], labels=["Young", "Middle", "Old"])

    AGE_counts = df.groupby("AGE_Group").size()
    diabetic_counts_AGE = df[df["CLASS"] == "DIABETIC"].groupby("AGE_Group").size()
    diabetic_counts_AGE = diabetic_counts_AGE.reindex(AGE_counts.index, fill_value=0)
    AGE_diabetes_percent = (diabetic_counts_AGE / AGE_counts * 100).fillna(0)

    print("\n% Diabetes by AGE Group:\n", AGE_diabetes_percent)

    plt.figure(figsize=(6,4))
    AGE_diabetes_percent.plot(kind='bar', color='orange')
    plt.title("Percentage of Diabetics by AGE Group")
    plt.ylabel("Percentage %")
    plt.tight_layout()
    plt.show()
else:
    AGE_diabetes_percent = pd.Series(dtype=float)
    print("\nAGE column missing — skipping age-based diabetes percentage.")

# ------------------------------------------------------
# 5. AGE distribution (diabetic vs non) - histogram
# ------------------------------------------------------
if 'AGE' in df.columns:
    plt.figure(figsize=(8,5))
    sns.histplot(data=df, x="AGE", hue="CLASS", kde=True, multiple='stack')
    plt.title("AGE Distribution: by CLASS")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------
# 6. Compare key health features by diabetes status (boxplots)
# ------------------------------------------------------
# use only features that exist
existing_key_features = [c for c in ["BMI", "HbA1c", "Chol", "TG", "Urea", "Cr"] if c in df.columns]
for col in existing_key_features:
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x="CLASS", y=col)
    plt.title(f"{col} Levels by CLASS")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------
# 7. Correlation Heatmap (numeric cols only)
# ------------------------------------------------------
num_df = df.select_dtypes(include=np.number)
if not num_df.empty:
    plt.figure(figsize=(12,8))
    sns.heatmap(num_df.corr(), annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap (numeric features)")
    plt.tight_layout()
    plt.show()
else:
    print("\nNo numeric columns available for correlation heatmap.")

# ------------------------------------------------------
# 8. Gender vs Class Heatmap (crosstab)
# ------------------------------------------------------
if 'Gender' in df.columns:
    matrix = pd.crosstab(df["Gender"], df["CLASS"])
    print("\nGender vs CLASS crosstab:\n", matrix)
    plt.figure(figsize=(6,4))
    sns.heatmap(matrix, annot=True, fmt='d', cmap="Purples")
    plt.title("Gender vs CLASS")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------
# 9. Pairplot for main features (only if columns exist and not too many rows)
# ------------------------------------------------------
pair_cols = [c for c in ["BMI","HbA1c","Chol","TG","AGE","CLASS"] if c in df.columns]
if len(pair_cols) >= 3 and df.shape[0] <= 2000:  # pairplot can be slow on large data
    sns.pairplot(df[pair_cols], hue="CLASS")
    plt.show()
else:
    print("\nSkipping pairplot (insufficient columns or too many rows).")

# ------------------------------------------------------
# 10. HbA1c vs BMI scatter (if both exist)
# ------------------------------------------------------
if "BMI" in df.columns and "HbA1c" in df.columns:
    plt.figure(figsize=(7,5))
    sns.scatterplot(data=df, x="BMI", y="HbA1c", hue="CLASS")
    plt.title("HbA1c vs BMI (Colored by CLASS)")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------
# 11. Summary Conclusions
# ------------------------------------------------------
print("\n" + "="*80)
print("SUMMARY OF KEY FINDINGS")
print("="*80)

print("\n1. Gender Diabetes Risk (percentages):")
print(gender_diabetes_percent if not gender_diabetes_percent.empty else "Not available")

print("\n2. AGE Group Diabetes Risk (percentages):")
print(AGE_diabetes_percent if not AGE_diabetes_percent.empty else "Not available")

print("\n3. Observed Strong Predictors (Visually):")
print("- HbA1c (if present) generally separates diabetic patients from others.")
print("- BMI (if present) tends to be higher for diabetic patients.")
print("- Chol, TG, Urea, Cr show differences across classes when present.")
print("- Older age groups typically show higher diabetes prevalence when AGE exists in dataset.")

print("\nEDA Completed.")
