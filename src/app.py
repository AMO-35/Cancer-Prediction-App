import joblib
import pandas as pd

# Sample input data 
sample_data = {
    'Age': 58,
    'Gender': 1,
    'BMI': 16.085313,
    'Smoking': 0,
    'GeneticRisk': 1,
    'PhysicalActivity': 8.146251,
    'AlcoholIntake': 4.148219,
    'CancerHistory': 1
}

# Convert to DataFrame
sample_data_df = pd.DataFrame([sample_data])

# Load the trained model (with preprocessing pipeline)
model = joblib.load('models/best_model_with_pipeline.pkl')

# Predict diagnosis
result = model.predict(sample_data_df)

# Output result
print("Diagnosis:", "Cancer" if result[0] == 1 else "No Cancer")
