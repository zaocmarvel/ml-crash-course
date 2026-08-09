#import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

#load dataset
diabetes_dataset = pd.read_csv('heart.csv')
training_df = diabetes_dataset

print('Read dataset completed successfully.')
print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
training_df.head(200)

#drop empty rows
training_df = training_df.dropna()

# View dataset statistics
print(training_df.shape)
print(training_df.isnull().sum())
print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
training_df.describe(include='all')

# View correlation matrix
training_df.corr(numeric_only = True)

input = training_df.iloc[:, :-1].values
output = training_df.iloc[:, -1].values

print(input.shape, output.shape)

model = Sequential()
model.add(Dense(1, input_dim=len(input[0, :]), activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(x=input, y=output, epochs=185, verbose=1)

for threshold in [0.5, 0.45, 0.4, 0.35, 0.3]:
    precision_metric = keras.metrics.Precision(thresholds=threshold)
    recall_metric = keras.metrics.Recall(thresholds=threshold)
    precision_metric.update_state(output, predictions)
    recall_metric.update_state(output, predictions)
    print(f"Threshold: {threshold} | Precision: {precision_metric.result():.3f} | Recall: {recall_metric.result():.3f}")
    
#loss curve
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss (Binary Crossentropy)')
plt.title('Training Loss Over Epochs')
plt.show()

# accuracy curve
plt.plot(history.history['accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Over Epochs')
plt.show()

#display preductions
predictions = model.predict(input)
predicted_labels = (predictions > 0.45).astype(int)

results = pd.DataFrame({
    "OBSERVED": output[:50],
    "PREDICTED_PROB": predictions[:50].flatten(),
    "PREDICTED_LABEL": predicted_labels[:50].flatten(),
})

header = "-" * 80
banner = header + "\n" + "|" + "PREDICTIONS".center(78) + "|" + "\n" + header
print(banner)
print(results)