from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
import os

app = Flask(__name__)

base_dir = os.path.dirname(os.path.abspath(__file__))

# Load model and preprocessing artifacts
model = pickle.load(open(os.path.join(base_dir, 'artifacts/model.pkl'), 'rb'))
preprocessor = pickle.load(open(os.path.join(base_dir, 'artifacts/preprocessor.pkl'), 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = float(request.form['age'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        gender = request.form['gender']
        hypertension = request.form['hypertension']
        heart_disease = request.form['heart_disease']
        ever_married = request.form['ever_married']
        work_type = request.form['work_type']
        residence_type = request.form['residence_type']
        smoking_status = request.form['smoking_status']

        # Create DataFrame
        data = [[age, avg_glucose_level, bmi, gender, hypertension, 
                heart_disease, ever_married, work_type, residence_type, smoking_status]]
        
        columns = ['age', 'avg_glucose_level', 'bmi', 'gender', 'hypertension', 
                  'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        
        data_df = pd.DataFrame(data, columns=columns)

        # Preprocess data
        pre_data = preprocessor.transform(data_df)
        
        # Make prediction
        prediction = model.predict(pre_data)
        probability = model.predict_proba(pre_data)[0][1]  # Probability of stroke

        # Format results
        result = "High Risk of Stroke" if prediction[0] == 1 else "Low Risk of Stroke"
        probability_percent = round(probability * 100, 2)

        return render_template('result.html', 
                            prediction=result,
                            probability=probability_percent,
                            input_data=request.form)

    except Exception as e:
        return render_template('error.html', error=str(e))
    
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)




