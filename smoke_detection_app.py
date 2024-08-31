import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('smoke_detection_iot.csv')  # Replace with your dataset path

# Drop missing values
df = df.dropna()

# Drop the 'Unnamed: 0' column
df = df.drop(['Unnamed: 0'], axis=1)

# Split dataset based on 'Fire Alarm'
smoke_detection_0 = df[df['Fire Alarm'] == 0]
smoke_detection_1 = df[df['Fire Alarm'] == 1]

[x_train_0, x_test_0, x_label_train_0, x_label_test_0] = train_test_split(smoke_detection_0, smoke_detection_0['Fire Alarm'], test_size=0.3, random_state=42, shuffle=True)
x_train_1_per = len(x_train_0) / len(smoke_detection_1)
[x_train_1, x_test_1, x_label_train_1, x_label_test_1] = train_test_split(smoke_detection_1, smoke_detection_1['Fire Alarm'], test_size=1 - x_train_1_per, random_state=42, shuffle=True)

x_train = pd.concat([x_train_0, x_train_1], axis=0)
x_test = pd.concat([x_test_0, x_test_1], axis=0)

x_train = x_train.drop(['Fire Alarm'], axis=1)
x_test = x_test.drop(['Fire Alarm'], axis=1)

x_train_label = np.array(pd.concat([x_label_train_0, x_label_train_1], axis=0))
x_test_label = np.array(pd.concat([x_label_test_0, x_label_test_1], axis=0))

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
model_dt.fit(x_train, x_train_label)
model_rf.fit(x_train, x_train_label)
model_nb.fit(x_train, x_train_label)

# Sidebar for user input
st.sidebar.header("Input Parameters")

def user_input_features():
    input_data = {}
    for feature in x_train.columns:
        input_data[feature] = st.sidebar.number_input(f"Input {feature}", value=float(x_train[feature].mean()))
    return pd.DataFrame(input_data, index=[0])

input_df = user_input_features()


# Main title
st.title("FireDetect AI")

# Model selection
st.header("Choose Your Model")
model_choice = st.selectbox("Select Model", ("FireGuard Ensemble", "Decision Tree", "Random Forest", "Naive Bayes"))

# Show Prediction
st.header("Prediction")

if model_choice == "FireGuard Ensemble":
    prediction_dt = model_dt.predict(input_df)
    prediction_rf = model_rf.predict(input_df)
    prediction_nb = model_nb.predict(input_df)
    
    # Majority vote for ensemble
    prediction = np.round((prediction_dt + prediction_rf + prediction_nb) / 3)
    prediction_proba = np.mean([model_dt.predict_proba(input_df), model_rf.predict_proba(input_df), model_nb.predict_proba(input_df)], axis=0)
    st.write("**Using FireGuard Ensemble**")
elif model_choice == "Random Forest":
    prediction = model_rf.predict(input_df)
    prediction_proba = model_rf.predict_proba(input_df)
    st.write("**Using Random Forest**")
elif model_choice == "Decision Tree":
    prediction = model_dt.predict(input_df)
    prediction_proba = model_dt.predict_proba(input_df)
    st.write("**Using Decision Tree**")
elif model_choice == "Naive Bayes":
    prediction = model_nb.predict(input_df)
    prediction_proba = model_nb.predict_proba(input_df)
    st.write("**Using Naive Bayes**")

if isinstance(prediction, (list, np.ndarray)):
    predicted_value = prediction[0]
else:
    predicted_value = prediction

if predicted_value == 1:
    st.write("Fire Alarm should ring!")
else:
    st.write("No need to ring the Fire Alarm.")

st.header("Prediction Probability")
st.write(prediction_proba)

# Show Model Accuracy
st.header("Model Accuracy")
if model_choice == "FireGuard Ensemble":
    accuracy_dt = accuracy_score(x_test_label, model_dt.predict(x_test))
    accuracy_rf = accuracy_score(x_test_label, model_rf.predict(x_test))
    accuracy_nb = accuracy_score(x_test_label, model_nb.predict(x_test))
    accuracy = np.mean([accuracy_dt, accuracy_rf, accuracy_nb])
elif model_choice == "Random Forest":
    accuracy = accuracy_score(x_test_label, model_rf.predict(x_test))
elif model_choice == "Decision Tree":
    accuracy = accuracy_score(x_test_label, model_dt.predict(x_test))
elif model_choice == "Naive Bayes":
    accuracy = accuracy_score(x_test_label, model_nb.predict(x_test))

st.write(f"Accuracy: {accuracy * 100:.2f}%")

# Feature Importances or Coefficients
st.header("Feature Importance")

if model_choice == "FireGuard Ensemble":
    importances_dt = model_dt.feature_importances_
    importances_rf = model_rf.feature_importances_
    importances_nb = np.abs(np.mean(model_nb.theta_, axis=0))  # Mean absolute value of Naive Bayes feature means

    # Normalize the importances so they can be averaged
    importances_dt /= np.sum(importances_dt)
    importances_rf /= np.sum(importances_rf)
    importances_nb /= np.sum(importances_nb)
    
    # Averaging the feature importances
    importances_avg = (importances_dt + importances_rf + importances_nb) / 3
    feature_importances = pd.Series(importances_avg, index=x_train.columns)
else:
    if model_choice == "Random Forest":
        feature_importances = pd.Series(model_rf.feature_importances_, index=x_train.columns)
    elif model_choice == "Decision Tree":
        feature_importances = pd.Series(model_dt.feature_importances_, index=x_train.columns)
    elif model_choice == "Naive Bayes":
        feature_importances = pd.Series(np.abs(np.mean(model_nb.theta_, axis=0)), index=x_train.columns)

# Sort the feature importances in descending order for better visualization
feature_importances = feature_importances.sort_values(ascending=False)

# Plotting the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Features")
st.pyplot(plt)
