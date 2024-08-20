import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('smoke_detection_iot.csv')  # Replace with your dataset path

# Drop the 'Unnamed: 0' column
df = df.drop(['Unnamed: 0'], axis=1)

# Define CSS for blue theme
blue_theme_css = """
    <style>
        body {
            background-color: #e3f2fd;  /* Light Blue Background */
            color: #0d47a1;  /* Dark Blue Text */
        }
        .main-title {
            font-family: 'Arial Black', sans-serif;
            color: #0d47a1;  /* Dark Blue */
            text-align: center;
            font-size: 36px;
        }
        .sub-title {
            font-family: 'Comic Sans MS', cursive, sans-serif;
            color: #90caf9;  /* Medium Light Blue */
            font-size: 24px;
        }
        .accuracy-text {
            font-family: 'Courier New', monospace;
            font-size: 20px;
            color: #0d47a1;  /* Dark Blue */
            background-color: #e3f2fd;  /* Light Blue Background */
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        .prediction-text {
            font-family: 'Verdana', sans-serif;
            font-size: 20px;
            color: #e3f2fd;  /* Light Blue Text */
            background-color: #0d47a1;  /* Dark Blue Background */
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        .model-box {
            background-color: #bbdefb;  /* Light Blue Background */
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            text-align: center;
        }
        .selectbox-container {
            margin-bottom: 20px;
        }
        .selectbox-container select {
            background-color: #bbdefb;  /* Light Blue Background */
            color: #0d47a1;  /* Dark Blue Text */
            border: 1px solid #0d47a1;
            border-radius: 8px;
            padding: 8px;
        }
    </style>
"""

# Apply the blue theme CSS
st.markdown(blue_theme_css, unsafe_allow_html=True)

# Main title
st.markdown("<h1 class='main-title'>Smoke Detection System</h1>", unsafe_allow_html=True)

# Display the dataset in the app
st.markdown("<h2 class='sub-title'>Dataset Preview</h2>", unsafe_allow_html=True)
st.write(df.head())

# Correlation analysis
st.markdown("<h2 class='sub-title'>Correlation Matrix</h2>", unsafe_allow_html=True)
correlation_matrix = df.corr()

# Visualize the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True)
st.pyplot(plt)

# Data Preprocessing
X = df.drop(['Fire Alarm'], axis=1)  # Features
y = df['Fire Alarm']  # Target variable

# Split the dataset (30% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data for Decision Tree and Random Forest
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
model_dt = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    ccp_alpha=0.01,
    random_state=42
)
model_rf = RandomForestClassifier(
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=10,
    max_features='sqrt',
    ccp_alpha=0.01,
    random_state=42
)
model_nb = GaussianNB()

# Train models
model_dt.fit(X_train_scaled, y_train)
model_rf.fit(X_train_scaled, y_train)
model_nb.fit(X_train, y_train)

# Sidebar for user input
st.sidebar.header("Input Parameters")

def user_input_features():
    input_data = {}
    for feature in X.columns:
        input_data[feature] = st.sidebar.number_input(f"Input {feature}", value=float(X[feature].mean()))
    return pd.DataFrame(input_data, index=[0])

input_df = user_input_features()

# Apply scaling to user input data for Decision Tree and Random Forest
input_df_scaled = scaler.transform(input_df)

# Model selection
st.markdown("<div class='model-box'><h1>Choose Your Model</h1></div>", unsafe_allow_html=True)
model_choice = st.selectbox("Select Model", ("Decision Tree", "Random Forest", "Naive Bayes"))

# Show Prediction
st.markdown("<h2 class='sub-title'>Prediction</h2>", unsafe_allow_html=True)

if model_choice == "Random Forest":
    prediction = model_rf.predict(input_df_scaled)
    prediction_proba = model_rf.predict_proba(input_df_scaled)
    st.write("**Using Random Forest**")
elif model_choice == "Decision Tree":
    prediction = model_dt.predict(input_df_scaled)
    prediction_proba = model_dt.predict_proba(input_df_scaled)
    st.write("**Using Decision Tree**")
elif model_choice == "Naive Bayes":
    prediction = model_nb.predict(input_df)
    prediction_proba = model_nb.predict_proba(input_df)
    st.write("**Using Naive Bayes**")

# Check if prediction is iterable
if isinstance(prediction, (list, np.ndarray)):
    predicted_value = prediction[0]
else:
    predicted_value = prediction

if predicted_value == 1:
    st.markdown("<p class='prediction-text'>Fire Alarm should ring!</p>", unsafe_allow_html=True)
else:
    st.markdown("<p class='prediction-text'>No need to ring the Fire Alarm.</p>", unsafe_allow_html=True)

st.markdown("<h2 class='sub-title'>Prediction Probability</h2>", unsafe_allow_html=True)
st.write(prediction_proba)

# Show Model Accuracy
st.markdown("<h2 class='sub-title'>Model Accuracy</h2>", unsafe_allow_html=True)
if model_choice == "Random Forest":
    accuracy = accuracy_score(y_test, model_rf.predict(X_test_scaled))
elif model_choice == "Decision Tree":
    accuracy = accuracy_score(y_test, model_dt.predict(X_test_scaled))
elif model_choice == "Naive Bayes":
    accuracy = accuracy_score(y_test, model_nb.predict(X_test))

st.markdown(f"<p class='accuracy-text'>Accuracy: {accuracy * 100:.2f}%</p>", unsafe_allow_html=True)

# Confusion Matrix
st.markdown("<h2 class='sub-title'>Confusion Matrix</h2>", unsafe_allow_html=True)
if model_choice == "Random Forest":
    cm = confusion_matrix(y_test, model_rf.predict(X_test_scaled))
elif model_choice == "Decision Tree":
    cm = confusion_matrix(y_test, model_dt.predict(X_test_scaled))
elif model_choice == "Naive Bayes":
    cm = confusion_matrix(y_test, model_nb.predict(X_test))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Fire', 'Fire'], yticklabels=['No Fire', 'Fire'])
plt.title('Confusion Matrix', fontsize=20, color='#0d47a1')
st.pyplot(plt)
