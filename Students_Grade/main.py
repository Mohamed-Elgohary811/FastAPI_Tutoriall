from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Enable CORS to allow the frontend to communicate with the FastAPI backend from different origins (domains, ports, or protocols)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class student(BaseModel):
    id: int
    name: str
    grade: int


students = [
    student(id=1, name="John Doe", grade=90),
    student(id=2, name="Jane Smith", grade=85)]

# read students
@app.get("/students/")
def read_students():
    return students

# create a new student
@app.post("/students/")
def create_student(new_student: student):
    students.append(new_student)
    return new_student

# update a student
@app.put("/students/{student_id}")
def update_student(student_id: int, updated_student: student):
    for index, student in enumerate(students):
        if student.id == student_id:
            students[index] = updated_student
            return updated_student
    return {"message": "Student not found"}


# delete a student
@app.delete("/students/{student_id}")
def delete_student(student_id: int):
    global students
    students = [student for student in students if student.id != student_id]
    return {"message": "Student deleted"}