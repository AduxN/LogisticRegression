import pandas as pd
import joblib
import ast
import sys


# Example new student's grades
# new_student_grades = [
#     [4, 3, 2],  # Grades for the AUS2 subject group (replace with actual grades)
#     [3, 4, 4],  # Grades for the OPTS subject group (replace with actual grades)
#     [5, 4, 3]   # Grades for the DIS subject group (replace with actual grades)
# ]

new_student_grades = [ast.literal_eval(sys.argv[1])]

model_files = ['model_AUS.pkl', 'model_OPTS.pkl', 'model_DIS.pkl']

# Create a list to store probabilities
probabilities = []

for i in range(3):
    new_student = new_student_grades[i]

    # Load the trained model from the file
    model = joblib.load(model_files[i])

    # Predict probability for the new student
    new_student_prob = model.predict_proba([new_student])[0, 1]

    # Store the probability
    probabilities.append(new_student_prob)

# Convert the list of probabilities to a DataFrame
probabilities_df = pd.DataFrame(probabilities, columns=['Probability'])
print(probabilities_df.values)
