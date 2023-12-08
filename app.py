from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np


app = Flask(__name__)
model = pickle.load(open('random_forest_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    data = request.form
    # Convert form data to the correct types
    try:
        glucose = float(data['glucose'])
        cholesterol = float(data['cholesterol'])
        hdl_chol = float(data['hdlChol'])
        age = float(data['age'])
        waist = float(data['waist'])
        systolic_bp = float(data['systolicBP'])
        weight = float(data['weight'])
        height = float(data['height'])
        diastolic_bp = float(data['diastolicBP'])
        hip = float(data['hip'])

        # Categorical data (ensure these match the form field names)
        gender = data['gender']
        physical_activity = data['physicalActivity']
        food_quality = data['foodQuality']
        family_history = data['familyHistory']
    except ValueError:
        return jsonify({'error': 'Invalid input data'}), 400

    # Calculate derived features
    chol_hdl_ratio = cholesterol / hdl_chol if hdl_chol != 0 else 0
    waist_hip_ratio = waist / hip if hip != 0 else 0
    bmi = weight / (height * height) * 703 

        # Encode categorical data as per training
    gender = data['gender']
    gender_female = 1 if gender == 'female' else 0
    gender_male = 1 if gender == 'male' else 0

    physical_activity = data['physicalActivity']
    physical_activity_high = 1 if physical_activity == 'High' else 0
    physical_activity_low = 1 if physical_activity == 'Low' else 0
    physical_activity_moderate = 1 if physical_activity == 'Moderate' else 0

    food_quality = data['foodQuality']
    food_quality_average = 1 if food_quality == 'Average' else 0
    food_quality_good = 1 if food_quality == 'Good' else 0
    food_quality_poor = 1 if food_quality == 'Poor' else 0

    family_history = data['familyHistory']
    family_history_no = 1 if family_history == 'No' else 0
    family_history_yes = 1 if family_history == 'Yes' else 0

        # Combine all features in the exact order as used during training
    features = [
            cholesterol, glucose, hdl_chol, chol_hdl_ratio, age, height,
            weight, bmi, systolic_bp, diastolic_bp, waist, hip,
            waist_hip_ratio, gender_female, gender_male,
            physical_activity_high, physical_activity_low,
            physical_activity_moderate, food_quality_average,
            food_quality_good, food_quality_poor, family_history_no,
            family_history_yes
        ]

    

    # Load your model (should be done outside this function in practice)
    # ...

    # Make a prediction
    prediction = model.predict([features])

    # Return the prediction result
    #return jsonify({'prediction': str(prediction), 'features': features})
    return render_template('result.html', prediction=prediction)


@app.route('/test')
def test():
    # Create a test input in the format your model expects
    # For example, a list of features
    test_input = [228.0,111.0,61.0,3.7,7.0,61.0,241.0,43.2,170.0,92.0,48.0,51.0,0.94,1,0,0,0,1,0,0,1,0,1]

    # Make a prediction
    prediction = model.predict([test_input])

    # Return the prediction result
    return jsonify({'test_prediction': str(prediction)})


if __name__ == "__main__":
    app.run(debug=True)