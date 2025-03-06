import pandas as pd

# Load the dataset
data = pd.read_csv("cleaned_file.csv")

# Fill missing values
data['rbc'].fillna('normal', inplace=True)
data['pc'].fillna('normal', inplace=True)
data['sod'].fillna(data['sod'].mean(), inplace=True)

# One-hot encode categorical variables
data = pd.get_dummies(data, columns=["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"])

# Separate features and target variable
X = data.drop("classification", axis=1)
y = data["classification"]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Make predictions
user_input = {
    'age': 48,
    'bp': 80,
    'sg': 1.02,
    'al': 1,
    'su': 0,
    'bgr': 121,
    'bu': 36,
    'sc': 1.2,
    'sod': data['sod'].mean(),  # Filled missing value with mean
    'pot': data['pot'].mean(),  # Assuming you want to fill this too
    'hemo': 15.4,
    'pcv': 44,
    'wc': 7800,
    'rc': 5.2,
    'htn_yes': 1,
    'htn_no': 0,
    'dm_yes': 1,
    'dm_no': 0,
    'cad_no': 1,
    'cad_yes': 0,
    'appet_good': 1,
    'appet_poor': 0,
    'pe_no': 1,
    'pe_yes': 0,
    'ane_no': 1,
    'ane_yes': 0
}

# Convert user input to DataFrame
user_df = pd.DataFrame([user_input])

# Make prediction
prediction = model.predict(user_df)
print("Prediction:", prediction)
