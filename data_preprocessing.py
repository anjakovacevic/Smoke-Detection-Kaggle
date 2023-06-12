import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
from imblearn.under_sampling import NearMiss

warnings.filterwarnings('ignore')

dataset = pd.read_csv("smoke_detection_iot.csv", index_col=0)
co = ['UTC', 'CNT']
dataset.drop(co, axis=1, inplace=True)

def display_initial_data():
    print(f"Shape Of The Dataset: {dataset.shape}")
    print(dataset.info())

    description = dataset.describe().transpose()
    description.to_csv("output.csv", index=True)

    fig, axs = plt.subplots(4, 3, figsize=(18, 12))

    sns.distplot(dataset['Humidity[%]'], ax=axs[0, 0])
    sns.distplot(dataset['Temperature[C]'], ax=axs[0, 1])
    sns.distplot(dataset['TVOC[ppb]'], ax=axs[0, 2])
    sns.distplot(dataset['eCO2[ppm]'], ax=axs[1, 0])
    sns.distplot(dataset['Raw Ethanol'], ax=axs[1, 1])
    sns.distplot(dataset['Pressure[hPa]'], ax=axs[1, 2])
    sns.distplot(dataset['Raw H2'], ax=axs[3, 2])
    sns.distplot(dataset['PM1.0'], ax=axs[3, 0])
    sns.distplot(dataset['PM2.5'], ax=axs[2, 2])
    sns.distplot(dataset['NC0.5'], ax=axs[2, 0])
    sns.distplot(dataset['NC1.0'], ax=axs[3, 1])
    sns.distplot(dataset['NC2.5'], ax=axs[2, 1])

    plt.tight_layout()


def target_value_ratio():
    fig, ax = plt.subplots()
    sns.set_theme(style="whitegrid")
    sns.countplot(x=dataset['Fire Alarm'])
    print("Dataset Shape:", dataset.shape)


def balance(X, y):
    near_miss = NearMiss(version=2, n_neighbors=3)
    X_balanced, y_balanced = near_miss.fit_resample(X, y)
    return X_balanced, y_balanced


def anomalies():
    X = dataset.drop(["Fire Alarm"], axis=1)
    y = dataset["Fire Alarm"]

    z_scores = np.abs((X - X.mean()) / X.std())
    # Set a threshold for Z-score 2.5 standard deviations
    threshold = 2.5

    outlier_indices = np.where(z_scores > threshold)

    cleaned_dataset = dataset.drop(outlier_indices[0])

    cleaned_X = cleaned_dataset.drop("Fire Alarm", axis=1)
    cleaned_y = cleaned_dataset["Fire Alarm"]

    print("Original Dataset Shape:", dataset.shape)
    print("Cleaned Dataset Shape:", cleaned_dataset.shape)

    fig, axs = plt.subplots(4, 3, figsize=(18, 12))

    sns.distplot(cleaned_dataset['Humidity[%]'], ax=axs[0, 0])
    sns.distplot(cleaned_dataset['Temperature[C]'], ax=axs[0, 1])
    sns.distplot(cleaned_dataset['TVOC[ppb]'], ax=axs[0, 2])
    sns.distplot(cleaned_dataset['eCO2[ppm]'], ax=axs[1, 0])
    sns.distplot(cleaned_dataset['Raw Ethanol'], ax=axs[1, 1])
    sns.distplot(cleaned_dataset['Pressure[hPa]'], ax=axs[1, 2])
    sns.distplot(cleaned_dataset['Raw H2'], ax=axs[3, 2])
    sns.distplot(cleaned_dataset['PM1.0'], ax=axs[3, 0])
    sns.distplot(cleaned_dataset['PM2.5'], ax=axs[2, 2])
    sns.distplot(cleaned_dataset['NC0.5'], ax=axs[2, 0])
    sns.distplot(cleaned_dataset['NC1.0'], ax=axs[3, 1])
    sns.distplot(cleaned_dataset['NC2.5'], ax=axs[2, 1])

    plt.tight_layout()
    return cleaned_dataset

def correlations():
    columns = ['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]', 'Raw H2',
               'Raw Ethanol', 'Pressure[hPa]', 'PM1.0', 'PM2.5', 'NC0.5', 'NC1.0', 'NC2.5']
    correlation_matrix = dataset[columns].corr()

    mask = np.zeros_like(correlation_matrix)
    mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, vmax=1, vmin=-1, annot=True, annot_kws={
        'fontsize': 7}, mask=mask, cmap=sns.diverging_palette(20, 220, as_cmap=True))

    # Displaying data dependencies; data selected based on the correlation matrix
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    dataset.plot.scatter(x='PM1.0', y='NC1.0', ax=axs[0, 0])    # 1
    dataset.plot.scatter(x='PM2.5', y='NC1.0', ax=axs[0, 1])    # 1
    dataset.plot.scatter(x='PM2.5', y='NC0.5', ax=axs[0, 2])    # 1
    dataset.plot.scatter(x='NC0.5', y='NC1.0', ax=axs[1, 0])    # 0.99
    dataset.plot.scatter(x='PM2.5', y='PM1.0', ax=axs[1, 1])    # 1
    dataset.plot.scatter(x='PM1.0', y='NC0.5', ax=axs[1, 2])    # 1
    plt.tight_layout()
    # Conclusion - remove PM1.0 and PM2.5

    # Checking dependencies without them
    columns_check = ['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]', 'Raw H2', 'Raw Ethanol',
                     'Pressure[hPa]', 'NC1.0', 'NC0.5', 'NC2.5']
    correlation_matrix_check = dataset[columns_check].corr()

    mask_check = np.zeros_like(correlation_matrix_check)
    mask_check[np.triu_indices_from(mask_check)] = True

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix_check, vmax=1, vmin=-1, annot=True, annot_kws={'fontsize': 7},
                mask=mask_check, cmap=sns.diverging_palette(20, 220, as_cmap=True), ax=ax2)

    plt.tight_layout()

    # Removing all attributes from cols
    cols = ['PM2.5', 'PM1.0']
    # cols = ['NC1.0', 'PM2.5', 'PM2.5']
    # cols = ['NC2.5.0', 'PM2.5', 'eCO2[ppm]']
    print("Deleting columns: ", cols)
    return cols


print("\n-------------------\nDisplaying initial data")
display_initial_data()
target_value_ratio()

# Anomalies
print("\n-------------------\nDeleting anomalies")
dataset = anomalies()

# Correlation matrix, removing dependent attributes
print("\n-------------------\nRemoving dependent attributes")
cols = correlations()
dataset.drop(cols, axis=1, inplace=True)

# Balancing the data
print("\n-------------------\nBalancing the data")
X = dataset.drop("Fire Alarm", axis=1)
y = dataset["Fire Alarm"]
X_balanced, y_balanced = balance(X, y)
dataset = pd.concat([X_balanced, y_balanced], axis=1)

# Seeing the balanced data
target_value_ratio()

ready_data = dataset
p = ready_data
p.to_csv("prepared.csv", index=True)

print("\nThe data is ready for algorithms!")

plt.show()


