from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent
model = joblib.load(ROOT / "model.joblib")

GRADES = {0: "Excellent", 1: "Good", 2: "Average", 3: "Below Average", 4: "Poor"}

app = FastAPI(title="Student Performance API")


@app.get("/")
async def home():
    return FileResponse(ROOT / "index.html")


@app.get("/predict")
async def predict(
    Age: int = Query(..., ge=15, le=18, description="Student age (15 to 18)"),
    Gender: int = Query(..., ge=0, le=1, description="0: Male, 1: Female"),
    Ethnicity: int = Query(..., ge=0, le=3, description="0: Caucasian, 1: African American, 2: Asian, 3: Other"),
    ParentalEducation: int = Query(..., ge=0, le=4, description="0: None ... 4: Higher"),
    StudyTimeWeekly: float = Query(..., ge=0, le=20, description="Weekly study hours (0 to 20)"),
    Absences: int = Query(..., ge=0, le=30, description="Absences (0 to 30)"),
    Tutoring: int = Query(..., ge=0, le=1, description="0: No, 1: Yes"),
    ParentalSupport: int = Query(..., ge=0, le=4, description="0: None ... 4: Very High"),
    Extracurricular: int = Query(..., ge=0, le=1, description="0: No, 1: Yes"),
    Sports: int = Query(..., ge=0, le=1, description="0: No, 1: Yes"),
    Music: int = Query(..., ge=0, le=1, description="0: No, 1: Yes"),
    Volunteering: int = Query(..., ge=0, le=1, description="0: No, 1: Yes"),
):
    x = np.array([[Age, Gender, Ethnicity, ParentalEducation, StudyTimeWeekly,
                   Absences, Tutoring, ParentalSupport, Extracurricular,
                   Sports, Music, Volunteering]])
    label = int(model.predict(x)[0])
    return {"predicted_grade": GRADES.get(label, "Unknown")}
