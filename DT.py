import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import plot_roc_curve

def show_decision_region(x, y, clf, f0, f1):
    plot_decision_regions(x, y, clf=clf)
    plt.xlabel(f0)
    plt.ylabel(f1)
    if clf.__class__.__name__ == "KNeighborsClassifier":
        plt.title(clf.__class__.__name__ + " k = " + str(clf.n_neighbors))
    else:
        plt.title(clf.__class__.__name__)
    plt.show()
    
data = pd.read_csv("wine_quality.csv")
data.info()

#attributes = data.iloc[:, 0:11].values
#quality = data.iloc[:, 11].values
attributes = data[["fixed.acidity","volatile.acidity","citric.acid","residual.sugar","chlorides","free.sulfur.dioxide","total.sulfur.dioxide","density","pH","sulphates","alcohol"]]
quality = data[["quality"]]

quality["quality"].value_counts()

# definição de classes e features
class_a = 5
class_b = 6
feature_0 = "alcohol"
feature_1 = "free.sulfur.dioxide"
feature_2 = "volatile.acidity"
feature_3 = "sulphates"
feature_4 = "density"

# filtrar classes e features
class_0_instances = (quality.values == class_a)
class_1_instances = (quality.values == class_b)

filtered_y = quality[class_0_instances | class_1_instances]
filtered_X = attributes[class_0_instances | class_1_instances]
# filtered_X = attributes
# filtered_X = filtered_X[[feature_0, feature_1,feature_2,feature_3,feature_4]]

X_train, X_test, y_train, y_test = train_test_split(filtered_X, filtered_y, test_size=0.1, random_state=199)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=199)

# X_train, X_test, y_train, y_test = train_test_split(attributes, quality, test_size=0.1, random_state=199)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=199)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# print(classification_report(y_test, model.predict(X_test)))

predicted_y = model.predict(X_test)

print(confusion_matrix(y_test, predicted_y)) #labels define como será a ordem das classes na matriz
plot_confusion_matrix(model, X_test, y_test)

print(classification_report(y_test, predicted_y))


# plt.figure()
# plot_tree(model, feature_names=["alcohol", "chlorides"])
print(classification_report(y_test, model.predict(X_test)))

# show_decision_region(
#     np.array(
#         [
#             X_test[feature_0].values, 
#             X_test[feature_1].values,
#             X_test[feature_2].values, 
#             X_test[feature_3].values,
#             X_test[feature_4].values 
#         ]
#     ).T, 
#     y_test["quality"].values, 
#     model, 
#     feature_0, 
#     feature_1,
#     feature_2, 
#     feature_3,
#     feature_4
# )

# plt.figure()
# plot_tree(model)

plot_roc_curve(model, X_test, y_test)