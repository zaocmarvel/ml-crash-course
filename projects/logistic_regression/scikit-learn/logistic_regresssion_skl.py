#import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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

X = training_df.iloc[:, :-1].values
y = training_df.iloc[:, -1].values

print(X.shape, y.shape)

#data split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

#create model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

  # probability of class 1
predicted_labels = (y_pred_proba > 0.45).astype(int)

print(confusion_matrix(y_test, predicted_labels))
print(classification_report(y_test, predicted_labels))

#display predictions
results = pd.DataFrame({
    "OBSERVED": y_test[:50],
    "PREDICTED_PROB": y_pred_proba[:50],
    "PREDICTED_LABEL": predicted_labels[:50],
})

header = "-" * 80
banner = header + "\n" + "|" + "PREDICTIONS".center(78) + "|" + "\n" + header
print(banner)
print(results)