# Load dependencies
import numpy as np
import pandas as pd
import keras
import ml_edu.experiment
import ml_edu.results
import plotly.express as px

# Load the dataset
housing_dataset = pd.read_csv("housing.csv")

# Updates dataframe to use specific columns.
training_df = housing_dataset #.loc[:, ('longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedroom', 'population', 'households', 'median_income', 'median_house_value', 'ocean_proximity')]

print('Read dataset completed successfully.')
print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
training_df.head(200)

training_df = training_df.dropna()
training_df = pd.get_dummies(training_df, columns=['ocean_proximity'])

# View dataset statistics
print(training_df.shape)
print(training_df.isnull().sum())
print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
training_df.describe(include='all')

# View correlation matrix
training_df.corr(numeric_only = True)

# Normalize the data using Z-scores ---
train_df_mean = training_df.mean(numeric_only=True)
train_df_std = training_df.std(numeric_only=True)
training_df_norm = (training_df[train_df_mean.index] - train_df_mean) / train_df_std

# Keep the original (non-normalized) label around so we can convert
# predictions back to real dollar values later
training_df_norm['median_house_value'] = training_df_norm['median_house_value']  # still normalized for training

def create_model(
    settings: ml_edu.experiment.ExperimentSettings,
    metrics: list[keras.metrics.Metric],
) -> keras.Model:
  inputs = {name: keras.Input(shape=(1,), name=name) for name in settings.input_features}
  concatenated_inputs = keras.layers.Concatenate()(list(inputs.values()))
  outputs = keras.layers.Dense(units=1)(concatenated_inputs)
  model = keras.Model(inputs=inputs, outputs=outputs)

  model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
                loss="mean_squared_error",
                metrics=metrics)
  return model


def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    label_name: str,
    settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:
  features = {name: dataset[name].values for name in settings.input_features}
  label = dataset[label_name].values
  history = model.fit(x=features,
                      y=label,
                      batch_size=settings.batch_size,
                      epochs=settings.number_epochs)

  return ml_edu.experiment.Experiment(
      name=experiment_name,
      settings=settings,
      model=model,
      epochs=history.epoch,
      metrics_history=pd.DataFrame(history.history),
  )

print("SUCCESS: defining linear regression functions complete.")

# Experiment 3
settings_3 = ml_edu.experiment.ExperimentSettings(
    learning_rate = 0.0002,
    number_epochs = 20,
    batch_size = 50,
    input_features = ['longitude', 'latitude', 'housing_median_age',
                       'rooms_per_household', 'bedrooms_per_room',
                       'population_per_household', 'median_income']
)

# NOTE: these derived features must be created BEFORE normalizing,
# so make sure this still happens on training_df, then re-normalize
training_df['rooms_per_household'] = training_df['total_rooms']/training_df['households']
training_df['bedrooms_per_room'] = training_df['total_bedrooms']/training_df['total_rooms']
training_df['population_per_household'] = training_df['population']/training_df['households']

# Re-run normalization now that derived columns exist
train_df_mean = training_df.mean(numeric_only=True)
train_df_std = training_df.std(numeric_only=True)
training_df_norm = (training_df[train_df_mean.index] - train_df_mean) / train_df_std

metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]

model_3 = create_model(settings_3, metrics)

experiment_3 = train_model('house_price_predictor', model_3, training_df_norm, 'median_house_value', settings_3)

ml_edu.results.plot_experiment_metrics(experiment_3, ['rmse'])


# --- Predictions: convert normalized output back to real dollars ---
def format_currency(x):
  return "${:.2f}".format(x)

def build_batch(df, batch_size):
  batch = df.sample(n=batch_size).copy()
  batch.set_index(np.arange(batch_size), inplace=True)
  return batch

def predict_fare(model, df, features, label, batch_size=50):
  batch = build_batch(df, batch_size)
  predicted_values = model.predict_on_batch(x={name: batch[name].values for name in features})

  data = {"PREDICTED_VALUE": [], "OBSERVED_VALUE": [], "L1_LOSS": []}
  for feature_name in features:
    data[feature_name] = []

  label_mean = train_df_mean[label]
  label_std = train_df_std[label]

  for i in range(batch_size):
    # un-normalize: real_value = (norm_value * std) + mean
    predicted = (predicted_values[i][0] * label_std) + label_mean
    observed = (batch.at[i, label] * label_std) + label_mean
    data["PREDICTED_VALUE"].append(format_currency(predicted))
    data["OBSERVED_VALUE"].append(format_currency(observed))
    data["L1_LOSS"].append(format_currency(abs(observed - predicted)))
    for feature_name in features:
      if isinstance(batch.at[i, feature_name], (int, float)):
        data[feature_name].append("{:.2f}".format(batch.at[i, feature_name]))
      else:
        data[feature_name].append(batch.at[i, feature_name])

  output_df = pd.DataFrame(data)
  return output_df

def show_predictions(output):
  header = "-" * 80
  banner = header + "\n" + "|" + "PREDICTIONS".center(78) + "|" + "\n" + header
  print(banner)
  print(output)
  return

# Predict using the NORMALIZED dataframe (since model was trained on normalized data)
output = predict_fare(experiment_3.model, training_df_norm, experiment_3.settings.input_features, 'median_house_value')
show_predictions(output)