import pandas as pd
import numpy as np

# Generate synthetic data for demonstration
np.random.seed(42)

# Number of users
num_users = 100

# Generate random user data
data = {
    'Age': np.random.randint(18, 60, num_users),
    'Gender': np.random.choice(['Male', 'Female'], num_users),
    'Weight': np.random.uniform(50, 100, num_users),
    'Height': np.random.uniform(150, 190, num_users),
    'Fitness_Goals': np.random.choice(['Weight Loss', 'Muscle Gain', 'General Fitness'], num_users),
    'Exercise_History': np.random.choice(['Beginner', 'Intermediate', 'Advanced'], num_users),
    'Medical_Conditions': np.random.choice(['None', 'Hypertension', 'Diabetes'], num_users)
}

# Create a DataFrame
user_df = pd.DataFrame(data)

# Display the generated dataset
print(user_df.head())

# Save the dataset to a CSV file
user_df.to_csv('user_data.csv', index=False)