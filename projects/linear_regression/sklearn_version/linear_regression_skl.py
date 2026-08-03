#modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#load in data
df = pd.read_csv("housing.csv")

training_df = df.dropna()
training_df = pd.get_dummies(training_df, columns=['ocean_proximity'])

training_df['rooms_per_household'] = training_df['total_rooms']/training_df['households']
training_df['bedrooms_per_room'] = training_df['total_bedrooms']/training_df['total_rooms']
training_df['population_per_household'] = training_df['population']/training_df['households']

training_df.describe(include='all')

# View correlation matrix
training_df.corr(numeric_only = True)

#initialze
X = training_df[['longitude', 'latitude', 'housing_median_age', 'rooms_per_household', 'bedrooms_per_room', 'population_per_household', 'median_income']].values
y = training_df['median_house_value'].values.reshape(-1, 1)
print(X.shape, y.shape)

#creating the model
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: ${rmse:,.2f}")

#plotting
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], c='r')  # perfect prediction line
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title('Predicted vs Observed House Value')
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
