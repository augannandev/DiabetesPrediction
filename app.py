from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(i) for i in request.form.values()]
    array_features = [np.array(features)]
    prediction = model.predict(array_features)
    
    output = prediction[0]
    
    return render_template('index.html', prediction_text='Diabetes Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
