from pydantic import BaseModel, EmailStr, Field  # Importing required classes from Pydantic
from typing import Optional  # Importing Optional for optional fields

# Define a Pydantic model for a Student
class Student(BaseModel):
    # Name is a required string with a default value of 'Avinash'
    name: str = 'Avinash'

    # Age is an optional integer
    age: Optional[int] = None

    # Email is a required field and must be a valid email string
    email: EmailStr

    # CGPA is a float between 0 and 10 (exclusive), default is 5
    cgpa: float = Field(
        gt=0, 
        lt=10, 
        default=5, 
        description='A decimal value representing the cgpa of the student'
    )

# Create a dictionary with some fields for the student
new_student = {
    'age': '28',  # Although age is a string here, Pydantic will auto-convert to int
    'email': 'avinashmzu101@gmail.com'
}

# Create a Student instance by unpacking the dictionary
student = Student(**new_student)

# Convert the Pydantic model to a regular dictionary
student_dict = dict(student)

# Print the age field from the dictionary
print(student_dict['age'])  # Output: 28

# Convert the model to a JSON string representation
student_json = student.model_dump_json()
