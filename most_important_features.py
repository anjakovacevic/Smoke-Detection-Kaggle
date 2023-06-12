import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, \
    f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv("smoke_detection_iot.csv", index_col=0)
columns = ['CNT', 'UTC']
dataset.drop(columns, axis=1, inplace=True)

y = dataset["Fire Alarm"]
X = dataset.drop(["Fire Alarm"], axis=1)

z_scores = np.abs((X - X.mean()) / X.std())
threshold = 2.5

outlier_indices = np.where(z_scores > threshold)
cleaned_dataset = dataset.drop(outlier_indices[0])
y = cleaned_dataset["Fire Alarm"]
cleaned_X = cleaned_dataset.drop("Fire Alarm", axis=1)

ros = RandomOverSampler(random_state=0)
X, y = ros.fit_resample(cleaned_X, y)

scaler = StandardScaler()

def rezultat(X, y, ime):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    models = [KNeighborsClassifier(), LogisticRegression(), DecisionTreeClassifier()]
    Name = []
    Accuracy = []
    F_score = []
    Precision = []
    Odziv = []
    Conf = []
    for model in models:
        Name.append(type(model).__name__)

        model.fit(x_train,y_train)
        prediction = model.predict(x_test)

        accuracyScore = round(100*accuracy_score(y_test, prediction), ndigits=2)
        Accuracy.append(accuracyScore)
        f1 = round(100*f1_score(y_test, prediction), ndigits=2)
        F_score.append(f1)
        pre = round(100*precision_score(y_test, prediction), ndigits=2)
        Precision.append(pre)
        odz = round(100*recall_score(y_test, prediction), ndigits=2)
        Odziv.append(odz)


        cm = confusion_matrix(y_test, prediction)
        Conf.append(cm[0])
        Conf.append(cm[1])
        Name.append(" ")
        Accuracy.append(" ")
        F_score.append(" ")
        Precision.append(" ")
        Odziv.append(" ")

    print("------------------------------------------\n", ime)
    Dict = {'Name': Name, 'Confusion Matrix': Conf, 'Accuracy': Accuracy, 'F score': F_score, 'Precision': Precision,
            'Recall': Odziv}
    model_df = pd.DataFrame(Dict)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(model_df)

# Results of all data
rezultat(X, y, "All data")


# PCA
param_grid = {
    'n_components': [7, 8, 4, 5, 6]
}

pca = PCA()
grid_search = GridSearchCV(estimator=pca, param_grid=param_grid, cv=5)
grid_search.fit(X, y)
best = grid_search.best_params_['n_components']
#print("Best component number of PCA = ", best)

pca = PCA(n_components=best)
pca.fit(X)
X_pca = pca.transform(X)
df_pca = pd.DataFrame(data=X_pca)
df_pca['Target'] = y

y_pca = df_pca['Target']
X_pca = df_pca.drop('Target', axis=1)

rezultat(X_pca, y, "PCA")

# Lasso
lasso = Lasso(alpha=0.02)
lasso.fit(X, y)
lasso_coefs = lasso.coef_
feature_names = X.columns

df_coefs = pd.DataFrame({'Feature': feature_names, 'Coefficient': lasso_coefs})

df_coefs = df_coefs.reindex(df_coefs['Coefficient'].abs().sort_values(ascending=False).index)
# print(df_coefs)
relevant_features = X.columns[lasso.coef_ != 0]
relevant_data = X[relevant_features]

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# print(relevant_data)

rezultat(relevant_data, y, "Lasso")







