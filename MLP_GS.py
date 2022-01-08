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
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import classification_report, plot_confusion_matrix
import seaborn as sns

import time

#%% Divisão da base de dados e pré-processamento
time_begin = time.time()

data = pd.read_csv("wine_quality.csv")
data.info()

attributes = data[["fixed.acidity","volatile.acidity","citric.acid","residual.sugar","chlorides","free.sulfur.dioxide","total.sulfur.dioxide","density","pH","sulphates","alcohol"]]
quality = data[["quality"]]

quality_dummy = np_utils.to_categorical(quality)

# attributes = data.iloc[:, 0:11].values
# quality = data.iloc[:, 11].values

X_train, X_test, y_train, y_test =  train_test_split(attributes, quality_dummy)

def createANN(epochs, activation, neurons):
    classifier = Sequential()
    classifier.add(Dense(units=neurons, activation=activation,input_dim=11))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(units=neurons, activation=activation))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(units=10, activation="softmax"))
    classifier.compile(optimizer="adam", loss="categorical_crossentropy",
                          metrics=["categorical_accuracy"])
    classifier.fit(X_train, y_train, batch_size=16,
                      epochs=epochs)
    return classifier

classifier = KerasClassifier(build_fn=createANN)

parameters={"epochs": [30,40],
            "activation": ["linear", "tanh"],
            "neurons": [4,8]}

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring="accuracy",
                           cv=2)

grid_search = grid_search.fit(X_train, y_train)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_

time_end = time.time()
print((time_end - time_begin)/60)