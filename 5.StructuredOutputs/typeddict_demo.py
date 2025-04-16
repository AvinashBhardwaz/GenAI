from typing import TypedDict  # Importing TypedDict from the typing module to define a dictionary with specific key-value types

# Defining a TypedDict for a Person with 'name' and 'age'
class Person(TypedDict):
    name: str  # The 'name' field will hold a string value
    age: int   # The 'age' field will hold an integer value

# Creating a new_person dictionary with the specified types for 'name' and 'age'
new_person: Person = {'name': 'Avinash', 'age': 28}  # Assigning values to 'name' (string) and 'age' (integer)

print(new_person)  # Output will be: {'name': 'Avinash', 'age': 28}
