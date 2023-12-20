import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
user_df = pd.read_csv('user_data.csv')

# Separate features (X) and target variable (y)
X = user_df.drop(columns=['Exercise'])
y = user_df['Exercise']

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X)

# Normalization of data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Feature selection using SelectKBest with ANOVA F-statistic
k_best = 5  # Set your desired number of features
feature_selector = SelectKBest(f_classif, k=k_best)
X_selected = feature_selector.fit_transform(X_normalized, y)

# Get the indices of the selected features
selected_feature_indices = feature_selector.get_support(indices=True)

# Get the names of the selected features
selected_feature_names = list(X.columns[selected_feature_indices])

# Display the selected features
print("\nSelected Feature Indices:", selected_feature_indices)
print("Selected Feature Names:", selected_feature_names)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train a machine learning model (example: RandomForestClassifier)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

#print("Test Set Shape:", X_test.shape)

# Make predictions
y_pred = model.predict(X_test)

print(y_pred)     #Note: Only 20 datasets' prediction of exercise was done as rest of the datasets were used for training the model

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)