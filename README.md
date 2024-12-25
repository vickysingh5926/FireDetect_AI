


# Fire Detection AI <br/>
A Machine Learning-based AI application for detecting smoke using IoT sensor data. This project is deployed on Streamlit to provide an interactive and user-friendly interface.

## Table of Contents <br/>
Introduction <br/> Features <br/> Tech Stack <br/> Installation <br/> Usage <br/> Project Structure <br/> Deployment <br/>

<hr>

## Introduction <br/>
This is a feature-rich project designed for detecting smoke using IoT sensor data. The application leverages Machine Learning models and is deployed on Streamlit for easy interaction and real-time results.
This project allows users to:

Detect smoke accurately using IoT sensor data. <br/>
View performance metrics like confusion matrix, accuracy, and feature importance. <br/>
Use an ensemble of models for improved predictions. <br/>
## Features <br/>
**Smoke Detection**: Accurately detects smoke using IoT sensor data. <br/>
**ML Models**: Utilizes Decision Tree, Random Forest, and Naive Bayes classifiers. <br/>
**Model Ensemble**: Includes an ensemble of multiple models for enhanced performance. <br/>
**User Interaction**: Provides a user-friendly interface powered by Streamlit. <br/>
**Performance Metrics**: Displays confusion matrix, accuracy score, and feature importance. <br/>
**Custom Dataset Splitting**: Splits datasets into separate classes before training to ensure balanced and accurate results. <br/>
**Real-time Results**: Offers quick and efficient smoke predictions. <br/>
## Tech Stack and Tools <br/>

**Streamlit** <br/>
**Backend & Machine Learning**: <br/>
**Python**<br/>
**Pandas** <br/>
**NumPy** <br/>
**Scikit-learn**<br/>
**Matplotlib**<br/>
## Installation <br/>
Follow these steps to set up the project locally:

## Prerequisites: <br/>
**Install Python (3.8 or later).** <br/>
**Install the required Python packages (see requirements.txt).** <br/>
## Steps: <br/>
## Clone the repository: <br/>

**git clone <repository_url>** 
**cd <project_folder>**  
## Install dependencies: <br/>

**pip install -r requirements.txt**  
## Run the Streamlit app: <br/>

**streamlit run smoke_detection_app.py**  
## Usage <br/>
Upload the IoT sensor dataset (smoke_detection_iot.csv) if required by the app. <br/>
Explore model performance metrics, including accuracy, confusion matrix, and feature importance. <br/>
Get predictions for smoke detection directly from the interface. <br/>
## Project Structure <br/>
**Fire_Detect (2).ipynb**: Jupyter notebook for training and testing ML models. <br/>
**smoke_detection_app.py**: Streamlit application for deploying the smoke detection system. <br/>
**requirements.txt:** Contains the necessary Python packages. <br/>
**smoke_detection_iot.csv**: Dataset used for training and testing the ML models. <br/>
**.gitignore**: Excludes environment files and other unnecessary files. <br/>
## Deployment <br/>
The application is deployed using Streamlit. Simply run the smoke_detection_app.py file locally or deploy it on any cloud platform that supports Streamlit for public access.


