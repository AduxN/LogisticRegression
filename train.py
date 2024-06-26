import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

# Load the CSV file
data = pd.read_csv('students_results_by_year.csv', sep=',')

# Encode the grades
grade_mapping = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1, 'Fx': 0}
data = data.map(lambda x: grade_mapping[x] if x in grade_mapping else np.nan)

# Drop rows with missing target subjects
data = data.dropna(subset=['Algoritmy a údajové štruktúry 2', 'Optimalizácia sietí', 'Diskrétna simulácia'])

# Define subject groups
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

# Initialize the Logistic Regression model
model = LogisticRegression()

model_files = ['model_AUS.pkl', 'model_OPTS.pkl', 'model_DIS.pkl']

# Train and save the models
for i in range(3):
    X = X_all[i]
    y = y_all[i]

    # Train the model on the entire dataset
    model.fit(X, y)

    # Save the trained model to a file
    joblib.dump(model, model_files[i])

print("Models trained and saved to pkl files.")
