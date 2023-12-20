import pandas as pd

# Exercise database
data = {
    'ExerciseName': ['Push-up', 'Squat', 'Bicep Curl', 'Deadlift', 'Plank'],
    'MuscleGroup': ['Chest', 'Legs', 'Arms', 'Back', 'Core'],
    'ExerciseType': ['lifting belts and knee wraps', 'Dumbbell', 'Barbell', 'exercise mat', 'exercise mat'],
    'DifficultyLevel': ['Intermediate', 'Advanced', 'Intermediate', 'Beginner', 'Beginner']
}

exercise_db = pd.DataFrame(data)

# Display the exercise database
print("Exercise Database:")
print(exercise_db)
