import streamlit as st
import pickle
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Math Performance Predictor",
    page_icon="🧮",
    layout="centered"
)

# Load model and encoders
@st.cache_resource
def load_model_and_encoders():
    """Load the trained model and encoders"""
    try:
        # Load model
        with open('models/mathe_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load encoders
        with open('models/encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        
        return model, encoders
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Define categories (from the original Flask app)
COUNTRIES = ['Ireland', 'Italy', 'Lithuania', 'Portugal', 'Romania', 'Russian Federation', 'Slovenia', 'Spain']
TOPICS = ['Analytic Geometry', 'Complex Numbers', 'Differential Equations', 'Differentiation', 
          'Fundamental Mathematics', 'Graph Theory', 'Integration', 'Linear Algebra', 
          'Numerical Methods', 'Optimization', 'Probability', 
          'Real Functions of a single variable', 'Set Theory', 'Statistics']
SUBTOPICS = ['Analytic Geometry', 'Complex Numbers', 'Differential Equations', 'Differentiation',
             'Fundamental Mathematics', 'Graph Theory', 'Integration', 'Numerical Methods',
             'Optimization', 'Probability', 'Real Functions of a single variable', 'Set Theory',
             'Statistics', 'Vector Spaces', 'Linear Systems', 'Matrices and Determinants',
             'Inner Product Spaces', 'Linear Transformation', 'Eigenvalues and Eigenvectors',
             'Orthogonality and Least Squares', 'Symmetric Matrices and Quadratic Forms',
             'The Singular Value Decomposition', 'Linear Transformation']
LEVELS = ['Advanced', 'Basic']

def main():
    """Main Streamlit application"""
    
    # Title
    st.title("🧮 Mathematics Performance Predictor")
    st.write("Predict whether a student will answer a mathematics question correctly")
    
    # Load model and encoders
    model, encoders = load_model_and_encoders()
    
    if model is None or encoders is None:
        st.error("Failed to load model. Please check that model files exist in the models/ directory.")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # Create input form
    with st.form("prediction_form"):
        st.subheader("� Student & Question Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            country = st.selectbox("Student Country", COUNTRIES)
            level = st.selectbox("Question Level", LEVELS)
        
        with col2:
            topic = st.selectbox("Mathematics Topic", TOPICS)
            subtopic = st.selectbox("Subtopic", SUBTOPICS)
        
        # Submit button
        submitted = st.form_submit_button("🎯 Predict Performance", type="primary")
        
        if submitted:
            # Make prediction
            try:
                # Encode the input features
                country_encoded = encoders['country'].transform([country])[0]
                level_encoded = encoders['level'].transform([level])[0]
                topic_encoded = encoders['topic'].transform([topic])[0]
                subtopic_encoded = encoders['subtopic'].transform([subtopic])[0]
                
                # Create feature vector
                features = np.array([[country_encoded, level_encoded, topic_encoded, subtopic_encoded]])
                
                # Make prediction
                prediction = model.predict(features)[0]
                probability = model.predict_proba(features)[0]
                
                # Get confidence score
                confidence = max(probability) * 100
                result = "Correct" if prediction == 1 else "Incorrect"
                
                # Display results
                st.divider()
                st.subheader("📊 Prediction Results")
                
                # Create columns for result display
                col1, col2 = st.columns(2)
                
                with col1:
                    if result == "Correct":
                        st.success(f"**Prediction: {result}** ✅")
                    else:
                        st.error(f"**Prediction: {result}** ❌")
                
                with col2:
                    st.info(f"**Confidence: {confidence:.1f}%**")
                
                # Progress bar for confidence
                st.write("**Confidence Level:**")
                st.progress(confidence/100)
                
                # Student profile summary
                with st.expander("👤 Student Profile Summary"):
                    st.write(f"**Country:** {country}")
                    st.write(f"**Question Level:** {level}")
                    st.write(f"**Topic:** {topic}")
                    st.write(f"**Subtopic:** {subtopic}")
                
                # Interpretation
                with st.expander("💡 Interpretation"):
                    if result == "Correct":
                        st.write(f"The model predicts this student is **likely to answer correctly** based on their profile. Students from {country} have shown specific performance patterns with {level.lower()} level {topic} questions.")
                    else:
                        st.write(f"The model predicts this student **may have difficulty** with this question. The model's analysis of students from {country} with similar {topic} questions suggests challenges with this particular combination.")
                    
                    st.write(f"The {confidence:.1f}% confidence level indicates {'high' if confidence >= 70 else 'medium' if confidence >= 55 else 'moderate'} model certainty in this prediction.")
            
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
    
    # Model information
    with st.expander("ℹ️ About This Model"):
        st.write("""
        **Model Details:**
        - **Algorithm:** Random Forest Classifier
        - **Accuracy:** ~60.5%
        - **Training Data:** 9,500+ student responses from European universities
        - **Features:** Country, Question Level, Topic, Subtopic
        
        **Feature Importance:**
        1. **Country** (51.1%) - Most influential factor
        2. **Subtopic** (26.1%) - Specific mathematical concept  
        3. **Topic** (15.6%) - General mathematical area
        4. **Level** (7.1%) - Question difficulty
        """)

if __name__ == "__main__":
    main()
