import time
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import warnings
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

dataset = pandas.read_csv("prepared.csv", index_col=0)

x = dataset.drop(["Fire Alarm"], axis=1)
y = dataset["Fire Alarm"]

# Scaling
sc = MinMaxScaler()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

print(f"Training set shape: {x_train.shape}, {y_train.shape}")
print(f"Test set shape: {x_test.shape}, {y_test.shape}")

models = [KNeighborsClassifier(), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]

Name = []
Accuracy = []
F_score = []
Precision = []
Recall = []
Time_Taken = []
Conf = []
Name2 = []
Best = []
New_Accuracy = []

for model in models:
    Name.append(type(model).__name__)
    begin = time.time()
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    end = time.time()

    accuracyScore = round(100 * accuracy_score(y_test, prediction), ndigits=2)
    Accuracy.append(accuracyScore)
    f1 = round(100 * f1_score(y_test, prediction), ndigits=2)
    F_score.append(f1)
    pre = round(100 * precision_score(y_test, prediction), ndigits=2)
    Precision.append(pre)
    rec = round(100 * recall_score(y_test, prediction), ndigits=2)
    Recall.append(rec)
    Time_Taken.append(end - begin)

    cm = confusion_matrix(y_test, prediction)
    Conf.append(cm[0])
    Conf.append(cm[1])
    Name.append(" ")
    Accuracy.append(" ")
    F_score.append(" ")
    Precision.append(" ")
    Recall.append(" ")
    Time_Taken.append(" ")

    if model.__class__.__name__ == 'LogisticRegression':
        hyperparameters = {
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear', 'saga'],
            'penalty': ['l1', 'l2', 'elasticnet']
        }
        grid = GridSearchCV(model, hyperparameters, cv=7)
        grid.fit(x_train, y_train)
        Best.append(grid.best_params_)
        model.C = grid.best_params_['C']
        model.solver = grid.best_params_['solver']
        model.penalty = grid.best_params_['penalty']
        model.fit(x_train, y_train)
        new_pred = model.predict(x_test)
        new_acc = round(100 * accuracy_score(y_test, new_pred), ndigits=2)
        Name2.append(model.__class__.__name__)
        New_Accuracy.append(new_acc)

    if model.__class__.__name__ == 'KNeighborsClassifier':
        hyperparameters = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        grid = GridSearchCV(model, hyperparameters, cv=7)
        grid.fit(x_train, y_train)
        Best.append(grid.best_params_)
        model.n_neighbors = grid.best_params_['n_neighbors']
        model.weights = grid.best_params_['weights']
        model.metric = grid.best_params_['metric']
        model.fit(x_train, y_train)
        new_pred = model.predict(x_test)
        new_acc = round(100 * accuracy_score(y_test, new_pred), ndigits=2)
        Name2.append(model.__class__.__name__)
        New_Accuracy.append(new_acc)

    if model.__class__.__name__ == 'DecisionTreeClassifier':
        hyperparameters = {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'splitter': ['best', 'random'],
            'ccp_alpha': [0.001, 0.1, 0.2, 0.5]
        }
        grid = GridSearchCV(model, hyperparameters, cv=7)
        grid.fit(x_train, y_train)
        Best.append(grid.best_params_)
        model.criterion = grid.best_params_['criterion']
        model.splitter = grid.best_params_['splitter']
        model.ccp_alpha = grid.best_params_['ccp_alpha']
        model.fit(x_train, y_train)
        new_pred = model.predict(x_test)
        new_acc = round(100 * accuracy_score(y_test, new_pred), ndigits=2)
        New_Accuracy.append(new_acc)
        Name2.append(model.__class__.__name__)

Dict = {'Name': Name, 'Confusion Matrix': Conf, 'Accuracy': Accuracy, 'F Score': F_score,
        'Precision': Precision, 'Recall': Recall, 'Time Taken': Time_Taken}
model_df = pandas.DataFrame(Dict)
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)
pandas.set_option('display.width', None)
pandas.set_option('display.max_colwidth', None)
print(model_df)

print("\nHyperparameter Tuning")
Dict_gscv = {'Name': Name2, 'Best Params': Best, 'New Accuracy': New_Accuracy}
model_gscv = pandas.DataFrame(Dict_gscv)

print(model_gscv)
