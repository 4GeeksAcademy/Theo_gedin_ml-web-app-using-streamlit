# ML Web App Using Streamlit

A simple and straightforward machine learning web application built with Streamlit for predicting student mathematics performance.

## Project Overview

This Streamlit web application predicts whether a student will answer a mathematics question correctly based on their profile and the question characteristics.

**Dataset:** [MathE Dataset for Assessing Mathematics Learning in Higher Education](https://archive.ics.uci.edu/dataset/1031/dataset+for+assessing+mathematics+learning+in+higher+education)

## What the App Does

**Input:**
- Student's Country (8 European countries)
- Question Level (Basic/Advanced)  
- Mathematics Topic (14 different topics)
- Subtopic (24 specific mathematical concepts)

**Output:**
- Prediction: Correct or Incorrect answer
- Confidence score (percentage)
- Visual confidence indicator
- Interpretation of the prediction

**Model:** Random Forest classifier trained on 9,500+ student responses from European universities, achieving 60.5% accuracy.

## Project Structure

```
├── app.py                      # Main Streamlit application
├── models/                     # Saved ML models
│   ├── mathe_model.pkl         # Trained Random Forest model
│   └── encoders.pkl            # Label encoders for categorical data
├── requirements.txt            # Python dependencies  
├── Procfile                    # Render deployment configuration
└── README.md                   # This file
```

## Usage

1. **Fill out the form:**
   - Select the student's country
   - Choose the question difficulty level  
   - Pick the mathematics topic and subtopic

2. **Get prediction:**
   - Click "Predict Performance" 
   - View the result with confidence score
   - Read the interpretation


## Model Information

- **Algorithm:** Random Forest Classifier
- **Features:** Country, Question Level, Topic, Subtopic  
- **Accuracy:** 60.5%
- **Training Data:** 9,548 student responses from European universities

**Feature Importance:**
1. Country (51.1%) - Most influential
2. Subtopic (26.1%) 
3. Topic (15.6%)
4. Level (7.1%)

## Why Streamlit?

- **Simple & Fast:** Quick development and deployment
- **Interactive:** Built-in widgets and real-time updates  
- **Clean UI:** Professional look without custom CSS
- **Easy Deployment:** Works great with Render, Heroku, Streamlit Cloud

## License

Educational project for the 4Geeks Academy ML bootcamp.