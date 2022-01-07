import pandas as pd
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import classification_report, plot_confusion_matrix
import seaborn as sns

data = pd.read_csv("wine_quality.csv")
data.info()

attributes = data[["fixed.acidity","volatile.acidity","citric.acid","residual.sugar","chlorides","free.sulfur.dioxide","total.sulfur.dioxide","density","pH","sulphates","alcohol"]]
quality = data[["quality"]]

# labelencoder = LabelEncoder()
# # quality = labelencoder.fit_transform(quality)
quality_dummy = np_utils.to_categorical(quality)

# attributes = data.iloc[:, 0:11].values
# quality = data.iloc[:, 11].values

X_train, X_test, y_train, y_test =  train_test_split(attributes, quality_dummy)

classifier = Sequential()
classifier.add(Dense(units=22, activation="linear",input_dim=11))
classifier.add(Dropout(0.1))
classifier.add(Dense(units=22, activation="linear"))
classifier.add(Dropout(0.1))
classifier.add(Dense(units=10, activation="softmax"))
classifier.compile(optimizer="adam", loss="categorical_crossentropy",
                      metrics=["categorical_accuracy"])
history = classifier.fit(X_train, y_train, batch_size=16,
                  epochs=50,validation_split=0.1)

forecasts = classifier.predict(X_test)

result =  classifier.evaluate(X_test, y_test)

y_test_transform = [np.argmax(t) for t in y_test]
forecasts_transform = [np.argmax(t) for t in forecasts]

precision = accuracy_score(y_test_transform, forecasts_transform)
print(precision)
matrix = confusion_matrix(forecasts_transform, y_test_transform)