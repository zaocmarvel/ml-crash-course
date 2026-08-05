#modules
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load in data
df = pd.read_csv("housing.csv")

training_df = df.dropna()
training_df = pd.get_dummies(training_df, columns=['ocean_proximity'])

training_df['rooms_per_household'] = training_df['total_rooms']/training_df['households']
training_df['bedrooms_per_room'] = training_df['total_bedrooms']/training_df['total_rooms']
training_df['population_per_household'] = training_df['population']/training_df['households']

#initialize
X = training_df[['longitude', 'latitude', 'housing_median_age', 'rooms_per_household', 'bedrooms_per_room', 'population_per_household', 'median_income']].values
y = training_df['median_house_value'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

#normalize
scaler = StandardScaler()
X_train_poly_scaled = scaler.fit_transform(X_train_poly)
X_test_poly_scaled = scaler.transform(X_test_poly)

print("Original features:", X_train.shape[1])
print("Polynomial features:", X_train_poly.shape[1])
print("Scaled features: ", X_train_poly_scaled.shape[1])

#creating the model
model = LinearRegression()
model.fit(X_train_poly_scaled, y_train)
y_pred = model.predict(X_test_poly_scaled)

#rmse
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: ${rmse:,.2f}")

#plotting
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], c='r')
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title('Predicted vs Observed House Value (Polynomial Features)')
plt.show()

#results
results = pd.DataFrame({
    "OBSERVED_VALUE": y_test.flatten(),
    "PREDICTED_VALUE": y_pred.flatten(),
})
results["L1_LOSS"] = (results["OBSERVED_VALUE"] - results["PREDICTED_VALUE"]).abs()
results["OBSERVED_VALUE"] = results["OBSERVED_VALUE"].map(lambda x: f"${x:,.2f}")
results["PREDICTED_VALUE"] = results["PREDICTED_VALUE"].map(lambda x: f"${x:,.2f}")
results["L1_LOSS"] = results["L1_LOSS"].map(lambda x: f"${x:,.2f}")

header = "-" * 80
banner = header + "\n" + "|" + "PREDICTIONS".center(78) + "|" + "\n" + header
print(banner)
print(results.sample(50))