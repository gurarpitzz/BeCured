import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the data
data = pd.read_csv('kidney_disease (1).csv')

# Handle missing values
data = data.fillna(method='ffill')  # Forward fill missing values

# Encode categorical variables using one-hot encoding
categorical_columns = ['htn', 'dm', 'cad', 'appet', 'pe', 'ane']
for column in categorical_columns:
    data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
    data.drop(column, axis=1, inplace=True)

# Drop the 'classification' column from X (features) and keep it in y (target)
X = data.drop('classification', axis=1)
y = data['classification']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Choose a model
model = RandomForestClassifier()

# Step 3: Train the model
model.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 5: Prediction (optional)
# You can now use the trained model to predict on new data.
