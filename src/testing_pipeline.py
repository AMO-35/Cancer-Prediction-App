import joblib
import pandas as pd

model = joblib.load('models/best_model_with_pipeline.pkl')

test_data = pd.read_csv('data/test.csv')

X_test = test_data.drop(columns=['Diagnosis'])
y_test = test_data['Diagnosis']

print(model.score(X_test, y_test))