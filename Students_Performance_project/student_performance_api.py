from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pycaret.classification import load_model, predict_model
import pandas as pd

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model('best_student_performance_model')

@app.get("/predict")
async def predict(
    Age: int = Query(
        ..., ge=15, le=18,
        description="Student age (15 to 18 years)"
    ),

    Gender: int = Query(
        ..., ge=0, le=1,
        description="Student gender (0: Female, 1: Male)"
    ),

    Ethnicity: int = Query(
        ..., ge=0, le=3,
        description="Student ethnicity (0: Caucasian, 1: African American, 2: Asian, 3: Other)"
    ),

    ParentalEducation: int = Query(
        ..., ge=0, le=4,
        description="Parent's education level"
    ),

    StudyTimeWeekly: float = Query(
        ..., ge=0, le=20,
        description="Weekly study time in hours (0 to 20)"
    ),

    Absences: int = Query(
        ..., ge=0, le=30,
        description="Number of absences during the school year (0 to 30)"
    ),

    Tutoring: int = Query(
        ..., ge=0, le=1,
        description="Tutoring (0: No, 1: Yes)"
    ),

    ParentalSupport: int = Query(
        ..., ge=0, le=4,
        description="Parental support level (0: None, 1: Low, 2: Moderate, 3: High, 4: Very High)"
    ),

    Extracurricular: int = Query(
        ..., ge=0, le=1,
        description="Participation in extracurricular activities (0: No, 1: Yes)"
    ),

    Sports: int = Query(
        ..., ge=0, le=1,
        description="Participation in sports (0: No, 1: Yes)"
    ),

    Music: int = Query(
        ..., ge=0, le=1,
        description="Participation in music activities (0: No, 1: Yes)"
    ),

    Volunteering: int = Query(
        ..., ge=0, le=1,
        description="Participation in volunteering (0: No, 1: Yes)"
    )
):
    data = {
        "Age": Age,
        "Gender": Gender,
        "Ethnicity": Ethnicity,
        "ParentalEducation": ParentalEducation,
        "StudyTimeWeekly": StudyTimeWeekly,
        "Absences": Absences,
        "Tutoring": Tutoring,
        "ParentalSupport": ParentalSupport,
        "Extracurricular": Extracurricular,
        "Sports": Sports,
        "Music": Music,
        "Volunteering": Volunteering
    }


    df = pd.DataFrame([data])

    predictions = predict_model(model, data=df)

    predicted_grade = predictions["prediction_label"].iloc[0]

    grade_map = {
        0: 'Excellent',
        1: 'Good',
        2: 'Average',
        3: 'Below Average',
        4: 'Poor'
    }

    grade = grade_map.get(int(predicted_grade), "Unknown")

    return {"predicted_grade": grade}


