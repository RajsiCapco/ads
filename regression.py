import pandas as pd
df = pd.read_csv(r"c:\Users\rmhs\OneDrive - Capco\Desktop\ADS\Rajsi ADS\insurance_regression.csv")
print(df.head())
df.info()
df. describe()


print(df["sex"].unique())
print(df["smoker"].unique())
print(df["region"].unique())

import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df["charges"], kde=True)
plt.title("Distribution of Charges")
plt.xlabel("Charges")
plt.ylabel("Frequency")
plt.show()

sns.boxplot(x="smoker", y="charges", data=df)
plt.title("Charges by Smoking Status")
plt.xlabel("Smoker")
plt.ylabel("Charges")
plt.show()

sns.scatterplot(x="bmi", y="charges", data=df)
plt.title("Charges by BMI")
plt.xlabel("bmi")
plt.ylabel("Charges")
plt.show()

sns.scatterplot(x="age", y="charges", data=df)
plt.title("Charges by Age")
plt.xlabel("age")
plt.ylabel("Charges")
plt.show()

sns.boxplot(x="children", y="charges", data=df)
plt.title("Charges by Children")
plt.xlabel("children")
plt.ylabel("Charges")
plt.show()

sns.scatterplot(x="age", y="charges", hue="smoker", data=df)
plt.title("Charges by Age and Smoking Status")
plt.xlabel("Age")
plt.ylabel("Charges")
plt.show()

sns.scatterplot(x="bmi", y="charges", hue="smoker", data=df)
plt.title("Charges by BMI and Smoking Status")
plt.xlabel("bmi")
plt.ylabel("Charges")
plt.show()

sns.boxplot(x="region", y="charges", hue="smoker", data=df)
plt.title("Charges by Region and Smoking Status")
plt.xlabel("region")
plt.ylabel("Charges")
plt.show()

sns.boxplot(x="children", y="bmi", hue="smoker", data=df)
plt.title("BMI by children and Smoking Status")
plt.xlabel("Children")
plt.ylabel("bmi")
plt.show()

corr=df.corr(numeric_only=True)

plt.figure(figsize=(8,6))
sns.heatmap(corr,annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

df_model = df.copy()
df_model["smoker"] = df_model["smoker"].map({"yes": 1, "no":0})
df_model["sex"] = df_model["sex"].map({"male":0, "female":1})
df_model = pd.get_dummies(df_model, columns=["region"], drop_first=True, dtype=int)

df_model.head()

df_model.dtypes
df_model.isnull().sum()

print(df_model.shape)
print(df_model.columns)

X = df_model.drop("charges", axis=1)
Y = df_model["charges"]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=13)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import numpy as np

def adjusted_r2 (r2, n, k):
    return 1-(1-r2)*(n-1)/(n-k-1)

n = X_test.shape[0]
k = X_test.shape[1]

mae_lr = mean_absolute_error(Y_test, Y_pred)
mse_lr = mean_squared_error(Y_test, Y_pred)
r2_lr = r2_score(Y_test,Y_pred)
rmse_lr = np.sqrt(mse_lr)
adj_r2_lr = adjusted_r2(r2_lr, n, k)

print("Linear Regression MAE:", mae_lr)
print("Linear Regression MSE:", mse_lr)
print("Linear Regression R2:", r2_lr)
print("Linear Regression RMSE:", rmse_lr)
print("Linear Regression Adj R2:", adj_r2_lr)


from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=13)

rf_model.fit(X_train, Y_train)
rf_pred = rf_model.predict(X_test)

mae_rf = mean_absolute_error(Y_test, rf_pred)
mse_rf = mean_squared_error(Y_test, rf_pred)
r2_rf = r2_score(Y_test, rf_pred)
rmse_rf = np.sqrt(mse_rf)
adj_r2_rf = adjusted_r2(r2_rf, n, k)

print("Random Forest MAE:", mae_rf)
print("Random Forest MSE:", mse_rf)
print("Random Forest R2:", r2_rf)
print("Random Forest RMSE:", rmse_rf)
print("Random Forest Adj R2:", adj_r2_rf)

import pandas as pd
comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"], 
    "MAE":[mae_lr, mae_rf], 
    "MSE":[mse_lr, mse_rf], 
    "r2": [r2_lr, r2_rf], 
    "RMSE": [rmse_lr, rmse_rf],
    "Adjusted R2": [adj_r2_lr, adj_r2_rf]
    })
print(comparison)

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.scatter(Y_test, Y_pred, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='red')
plt.title("Linear Regression: Actual vs Predicted")
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")

plt.subplot(1,2,2)
plt.scatter(Y_test, rf_pred, alpha=0.5, color='green')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='red')
plt.title("Random Forest: Actual vs Predicted")
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")

plt.tight_layout()
plt.show()

importances = pd.DataFrame({
    'Features': X.columns,
    'Importance': rf_model.feature_importances_
})

importances = importances.sort_values(by="Importance", ascending = False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Features", data=importances)
plt.title("Random Forest Features Importances")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()