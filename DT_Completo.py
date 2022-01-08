import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
    
data = pd.read_csv("wine_quality.csv")
data.info()

attributes = data[["fixed.acidity","volatile.acidity","citric.acid","residual.sugar","chlorides","free.sulfur.dioxide","total.sulfur.dioxide","density","pH","sulphates","alcohol"]]
quality = data[["quality"]]

quality["quality"].value_counts()

X_train, X_test, y_train, y_test = train_test_split(attributes, quality, test_size=0.1, random_state=199)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=199)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# print(classification_report(y_test, model.predict(X_test)))

# plt.figure()
# plot_tree(model)

predicted_y = model.predict(X_test)
label = [0, 1]
print(confusion_matrix(y_test, predicted_y)) #labels define como será a ordem das classes na matriz
plot_confusion_matrix(model, X_test, y_test)

plot_roc_curve(model, X_test, y_test)