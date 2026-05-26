from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('loan_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from the JSON request
        data = request.get_json()
        
        income = float(data['monthly_income'])
        age = float(data['age'])
        cibil = float(data['cibil_score'])
        existing_loan = float(data['existing_loan'])
        
        # Format and scale features
        features = np.array([[income, age, cibil, existing_loan]])
        scaled_features = scaler.transform(features)
        
        # Perform prediction
        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]
        confidence = round(probabilities[prediction] * 100, 2)
        
        result = "Approved" if prediction == 1 else "Rejected"
        
        return jsonify({
            'status': 'success',
            'prediction': result,
            'confidence': f"{confidence}%"
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)