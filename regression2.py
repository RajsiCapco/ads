# %%
#Import in all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# %%
#Import Insurance Regression Data and describe the data
df = pd.read_csv(r"c:\Users\rmhs\OneDrive - Capco\Desktop\ADS\Rajsi ADS\Regression\data\insurance_regression.csv")
print(df.head(10))
df.info()
df. describe()

# %%
#Feature Engineering to ensure all data is model-ready
# Change all object *text* data to numerical - note variables are categorical
print(df["sex"].unique())
print(df["smoker"].unique())
print(df["region"].unique())

# %%
#rename variable sex to gender
df = df.rename(columns={"sex": "gender"})

# %%
#create copy of data
df_change = df.copy()

#convert smoker and sex - binary categorical- to numerical 1,0
df_change["smoker"] = df_change["smoker"].map({"yes": 1, "no":0})
df_change["gender"] = df_change["gender"].map({"male":0, "female":1})
#convert region to nominal numerical with label encoding - not one-hot encoding (to ensure everything stays in one column)
df_change["region_encoded"] = le.fit_transform(df_change["region"])
#drop the original text region - not going to be used for model
df_2 = df_change.drop("region", axis=1)


# %%
# rename variable sex to gender and region_encoded to region
df_2 = df_2.rename(columns={"region_encoded": "region"})

# %%
df_2.info()
df_2.describe()

# %%
#EDA - Exploratory Feature Analysis - used to understand data structure and relationships
# Charges is Target Variable

#Target variable distribution
sns.histplot(df["charges"], kde=True)
plt.title("Distribution of Charges")
plt.xlabel("Charges")
plt.ylabel("Frequency")
plt.show()

# %%
# Running all INDIVIDUAL variables against charges to see which has most impact against target variable
# Set figure size and layout
fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 2 rows, 3 columns
plt.tight_layout(pad=4.0)

# Plot 1: Smoker
sns.boxplot(x="smoker", y="charges", data=df, ax=axes[0, 0])
axes[0, 0].set_title("Charges by Smoking Status")

# Plot 2: Age
sns.scatterplot(x="age", y="charges", data=df, ax=axes[0, 1])
axes[0, 1].set_title("Charges by Age")

# Plot 3: Gender
sns.boxplot(x="gender", y="charges", data=df, ax=axes[0, 2])
axes[0, 2].set_title("Charges by Gender")

# Plot 4: Children
sns.boxplot(x="children", y="charges", data=df, ax=axes[1, 0])
axes[1, 0].set_title("Charges by Children")

# Plot 5: BMI
sns.scatterplot(x="bmi", y="charges", data=df, ax=axes[1, 1])
axes[1, 1].set_title("Charges by BMI")

# Plot 6: Region
sns.boxplot(x="region", y="charges", data=df, ax=axes[1, 2])
axes[1, 2].set_title("Charges by Region")

# Show all plots at once
plt.show()

# %%
# Select 6 features (adjust as needed)
features = ["age", "bmi", "smoker", "children", "region", "gender"]
n = len(features)

fig, axes = plt.subplots(n, n, figsize=(20, 20))
plt.subplots_adjust(hspace=0.5, wspace=0.4)

for i in range(n):
    for j in range(n):
        ax = axes[i, j]
        x = features[j]
        hue = features[i]

        # Diagonal: plot x vs charges (alone)
        if i == j:
            if df[x].dtype == "object" or df[x].nunique() < 10:
                sns.boxplot(x=x, y="charges", data=df, ax=ax)
            else:
                sns.scatterplot(x=x, y="charges", data=df, ax=ax)
            ax.set_title(f"{x} vs Charges")
        
        # Off-diagonal: interaction plots
        else:
            if df[hue].dtype == "object" or df[hue].nunique() < 10:
                sns.scatterplot(x=x, y="charges", hue=hue, data=df, ax=ax, legend=False)
            else:
                sns.scatterplot(x=x, y="charges", size=hue, data=df, ax=ax, legend=False)
            ax.set_title(f"{x} vs Charges by {hue}")

        ax.set_xlabel("")
        ax.set_ylabel("")

plt.suptitle("EDA Grid: Feature Interactions and Impact on Charges", fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


# %%
#basic heatmap with target correlation
# Compute correlation matrix (only numeric columns)
corr = df_2.corr(numeric_only=True)

# Sort by correlation with target (optional but useful)
corr_target = corr["charges"].sort_values(ascending=False)

# Plot full heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (df_2 - Encoded & Engineered)")
plt.show()

# Plot only correlations with charges
plt.figure(figsize=(8, 6))
sns.barplot(x=corr_target.values, y=corr_target.index)
plt.title("Correlation with Charges")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Feature")
plt.show()

corr = df_2.corr(numeric_only=True)
corr["charges"].sort_values(ascending=False)

# %%
#EDA shows that Smoker is a huge impact on charges - drop smoker from the dataset and see if anything else has a strong interaction

df_nosmoker = df_2.drop("smoker", axis=1)

#basic heatmap with target correlation
# Compute correlation matrix (only numeric columns)
corr_nonsmoker = df_nosmoker.corr(numeric_only=True)

# Sort by correlation with target (optional but useful)
corr_targetns = corr_nonsmoker["charges"].sort_values(ascending=False)

# Plot full heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_nonsmoker, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (df_nosmoker)")
plt.show()

# Plot only correlations with charges
plt.figure(figsize=(8, 6))
sns.barplot(x=corr_targetns.values, y=corr_targetns.index)
plt.title("Correlation with Charges")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Feature")
plt.show()

corr_ns = df_nosmoker.corr(numeric_only=True)
corr_ns["charges"].sort_values(ascending=False)

# %%
df_model = df_2.copy()
print(df_model.shape)
print(df_model.columns)

# %%
# A: Full feature set (including 'smoker')
X_full = df_model.drop("charges", axis=1)

# B: Reduced feature set (excluding 'smoker' and optionally 'bmi_smoker' if it exists)
X_nosmoker = df_model.drop(["charges", "smoker"], axis=1)

# Target
y = df_model["charges"]

# %%
# Full version
X_train_full, X_test_full, y_train, y_test = train_test_split(X_full, y, test_size=0.3, random_state=13)

# Reduced version
X_train_nosmoker, X_test_nosmoker, y_train_nosmoker,y_test_nosmoker = train_test_split(X_nosmoker, y, test_size=0.3, random_state=13)

# %%
# Model A: With smoker
rf_full = RandomForestRegressor(random_state=13)
rf_full.fit(X_train_full, y_train)
pred_full = rf_full.predict(X_test_full)

# Model B: Without smoker
rf_nosmoker = RandomForestRegressor(random_state=13)
rf_nosmoker.fit(X_train_nosmoker, y_train_nosmoker)
pred_nosmoker = rf_nosmoker.predict(X_test_nosmoker)

# %%
print("RF Model A (all features):")
print("R²:", r2_score(y_test, pred_full))
print("MAE:", mean_absolute_error(y_test, pred_full))

print("RF Model B (without smoker):")
print("R²:", r2_score(y_test, pred_nosmoker))
print("MAE:", mean_absolute_error(y_test_nosmoker, pred_nosmoker))

# %%
importances_full = pd.Series(rf_full.feature_importances_, index=X_full.columns)
importances_nosmoker = pd.Series(rf_nosmoker.feature_importances_, index=X_nosmoker.columns)

# Side-by-side comparison
comparison_df = pd.DataFrame({
    "With Smoker": importances_full,
    "Without Smoker": importances_nosmoker
}).sort_values("With Smoker", ascending=False)

print(comparison_df)

# %%
plt.figure(figsize=(10, 6))
comparison_df.plot(kind="barh", figsize=(10, 6), width=0.8)

plt.title("Feature Importance Comparison: With vs Without Smoker")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# %%
r2_full = r2_score(y_test, pred_full)
r2_nosmoker = r2_score(y_test_nosmoker, pred_nosmoker)

mae_full = mean_absolute_error(y_test, pred_full)
mae_nosmoker = mean_absolute_error(y_test_nosmoker, pred_nosmoker)

sns.set(style="whitegrid")
plt.figure(figsize=(12, 5))

# --- Plot 1: With Smoker ---
ax1 = plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=pred_full, alpha=0.5, color='blue', ax=ax1)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title("Model A: With Smoker", fontsize=12)
plt.xlabel("Actual Charges", fontsize=10)
plt.ylabel("Predicted Charges", fontsize=10)
plt.text(y_test.min(), y_test.max()*0.95, f"R²: {r2_full:.2f}\nMAE: {mae_full:.0f}", fontsize=9)

# --- Plot 2: Without Smoker ---
ax2 = plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test, y=pred_nosmoker, alpha=0.5, color='green', ax=ax2)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title("Model B: Without Smoker", fontsize=12)
plt.xlabel("Actual Charges", fontsize=10)
plt.ylabel("Predicted Charges", fontsize=10)
plt.text(y_test.min(), y_test.max()*0.95, f"R²: {r2_nosmoker:.2f}\nMAE: {mae_nosmoker:.0f}", fontsize=9)

plt.suptitle("Actual vs Predicted Charges — Random Forest Model Comparison", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# %%



