import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

# Load the CSV file
data = pd.read_csv('students_results_by_year.csv', sep=',')

# Encode the grades
grade_mapping = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1, 'Fx': 0}
data = (data.map(lambda x: grade_mapping[x] if x in grade_mapping else np.nan)
        .dropna(subset=['Algoritmy a údajové štruktúry 2'])
        .dropna(subset=['Optimalizácia sietí'])
        .dropna(subset=['Diskrétna simulácia']))

aus2_subjects = ['Algoritmy a údajove štruktúry 1', 'Informatika 3', 'Informatika 2', 'Algoritmy a údajové štruktúry 2']
opts_subjects = ['Matematika pre informatikov', 'Algoritmická teória grafov', 'Diskrétna optimalizácia', 'Optimalizácia sietí']
dis_subjects = ['Diskrétna pravdepodobnosť', 'Pravdepodobnosť a štatistika', 'Modelovanie a simulácia', 'Diskrétna simulácia']

dataAUS = data[aus2_subjects].copy()
dataOPTS = data[opts_subjects].copy()
dataDIS = data[dis_subjects].copy()

# Define the target variable (pass/fail)
dataAUS['pass_fail'] = dataAUS.apply(lambda row: 1 if row['Algoritmy a údajové štruktúry 2'] > 0 else 0, axis=1)
dataOPTS['pass_fail'] = dataOPTS.apply(lambda row: 1 if row['Optimalizácia sietí'] > 0 else 0, axis=1)
dataDIS['pass_fail'] = dataDIS.apply(lambda row: 1 if row['Diskrétna simulácia'] > 0 else 0, axis=1)


# Split the data into features and target
X_AUS = dataAUS.drop(columns=['Algoritmy a údajové štruktúry 2', 'pass_fail'])
y_AUS = dataAUS['pass_fail']
X_OPTS = dataOPTS.drop(columns=['Optimalizácia sietí', 'pass_fail'])
y_OPTS = dataOPTS['pass_fail']
X_DIS = dataDIS.drop(columns=['Diskrétna simulácia', 'pass_fail'])
y_DIS = dataDIS['pass_fail']

X_all = [X_AUS, X_OPTS, X_DIS]
y_all = [y_AUS, y_OPTS, y_DIS]

model = LogisticRegression()

# Initialize Stratified K-Fold
skf = StratifiedKFold(n_splits=5)

# Define the threshold values
thresholds = np.arange(0.1, 1.0, 0.1)

# Create a dataframe to store the results
results = pd.DataFrame(
    columns=['t', 'T_Total', 'T_Specificita', 'T_Senzitivita', 'T_Priemer', 'V_Total',
             'V_Specificita', 'V_Senzitivita', 'V_Priemer'])


# Function to calculate specificity
def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)


for i in range(3):
    X = X_all[i]
    y = y_all[i]

    # Loop over thresholds
    for threshold in thresholds:
        train_accuracies = []
        train_specificities = []
        train_recalls = []
        train_means = []
        val_accuracies = []
        val_specificities = []
        val_recalls = []
        val_means = []

        # Perform Stratified K-Fold cross-validation
        for train_index, val_index in skf.split(X, y):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            # Train the model
            model.fit(X_train, y_train)

            # Predict probabilities
            y_train_pred_prob = model.predict_proba(X_train)[:, 1]
            y_val_pred_prob = model.predict_proba(X_val)[:, 1]

            # Apply threshold
            y_train_pred = (y_train_pred_prob >= threshold).astype(int)
            y_val_pred = (y_val_pred_prob >= threshold).astype(int)

            # Calculate metrics for train
            train_accuracies.append(accuracy_score(y_train, y_train_pred))
            train_specificities.append(specificity_score(y_train, y_train_pred))
            train_recalls.append(recall_score(y_train, y_train_pred))
            train_means.append((specificity_score(y_train, y_train_pred) + recall_score(y_train, y_train_pred)) / 2)

            # Calculate metrics for validation
            val_accuracies.append(accuracy_score(y_val, y_val_pred))
            val_specificities.append(specificity_score(y_val, y_val_pred))
            val_recalls.append(recall_score(y_val, y_val_pred))
            val_means.append((specificity_score(y_val, y_val_pred) + recall_score(y_val, y_val_pred)) / 2)

        # Append results to the dataframe
        results = results._append({
            't': threshold,
            'T_Total': np.mean(train_accuracies),
            'T_Specificita': np.mean(train_specificities),
            'T_Senzitivita': np.mean(train_recalls),
            'T_Priemer': np.mean(train_means),
            'V_Total': np.mean(val_accuracies),
            'V_Specificita': np.mean(val_specificities),
            'V_Senzitivita': np.mean(val_recalls),
            'V_Priemer': np.mean(val_means)
        }, ignore_index=True)

    results.to_csv(f'results_{i}.csv', index=False)
    results = pd.DataFrame(
    columns=['t', 'T_Total', 'T_Specificita', 'T_Senzitivita', 'T_Priemer', 'V_Total',
             'V_Specificita', 'V_Senzitivita', 'V_Priemer'])
