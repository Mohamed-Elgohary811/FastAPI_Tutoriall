from pycaret.classification import *
import pandas as pd

data = pd.read_csv('Student_performance_dataset.CSV')
data.head()

# Initializes the PyCaret environment and prepares the dataset for machine learning
clf = setup(
    data,
    target='GradeClass',
    session_id=123,
    numeric_features=['Age','StudyTimeWeekly','Absences'],
    categorical_features=['Gender','Ethnicity','ParentalEducation','Tutoring','ParentalSupport','Extracurricular','Sports','Music','Volunteering'],
    ignore_features=['StudentID', 'GPA']
)


best_model = compare_models()

save_model(best_model, 'best_student_performance_model')
print("Best model saved")