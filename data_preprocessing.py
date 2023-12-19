import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# Load the dataset
user_df = pd.read_csv('user_data.csv')

# Display basic information about the dataset
print("Original Dataset Information:")
print(user_df.info())

# Check for missing values
print("\nMissing Values:")
print(user_df.isnull().sum())

# Handle missing values by filling NaN values with the mean for numeric columns
numeric_columns = user_df.select_dtypes(include=['float64', 'int64']).columns
user_df[numeric_columns] = user_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
user_df[numeric_columns] = user_df[numeric_columns].fillna(user_df[numeric_columns].mean())

# Display statistics of numerical features
print("\nNumerical Feature Statistics:")
print(user_df.describe())

# Convert categorical features to numerical using one-hot encoding
user_df = pd.get_dummies(user_df, columns=['Gender', 'Fitness_Goals', 'Exercise_History', 'Medical_Conditions'], drop_first=True)

# Display the modified dataset
print("\nModified Dataset:")
print(user_df.head())

# Normalization of data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(user_df)

# Feature selection
# Select top k features based on ANOVA F-statistic (you can adjust k based on your needs)
k_best = SelectKBest(f_regression, k='all')
X_selected = k_best.fit_transform(X_normalized, user_df['Fitness_Goals'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, user_df['Fitness_Goals'], test_size=0.2, random_state=42)

# Display the selected features (optional)
selected_feature_names = user_df.columns[k_best.get_support(indices=True)].tolist()
print("\nSelected Features:", selected_feature_names)
