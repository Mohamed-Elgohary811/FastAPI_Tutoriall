import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

FEATURES = ["Age", "Gender", "Ethnicity", "ParentalEducation", "StudyTimeWeekly",
            "Absences", "Tutoring", "ParentalSupport", "Extracurricular",
            "Sports", "Music", "Volunteering"]

data = pd.read_csv("Student_performance_dataset.CSV")
X = data[FEATURES].values
y = data["GradeClass"].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y)

model = RandomForestClassifier(n_estimators=200, random_state=123, n_jobs=-1)
model.fit(X_train, y_train)
print("Test accuracy:", round(accuracy_score(y_test, model.predict(X_test)), 4))

# Final model uses all the data
model.fit(X, y)
joblib.dump(model, "model.joblib", compress=3)
print("Saved model.joblib")
