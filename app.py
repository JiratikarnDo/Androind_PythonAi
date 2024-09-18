from flask import Flask, request, jsonify
import joblib
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the model
model = joblib.load("heart_Model_Ai.pkl")

@app.route('/api/heart', methods=['POST'])
def house():
    age = int(request.form.get('age')) 
    sex = int(request.form.get('sex')) 
    cp = int(request.form.get('cp')) 
    trestbps = int(request.form.get('trestbps')) 
    chol = int(request.form.get('chol'))
    thalach = int(request.form.get('thalach'))
    oldpeak = float(request.form.get('oldpeak'))
    
    # Prepare the input for the model
    x = np.array([[age, sex, cp, trestbps, chol, thalach, oldpeak]])

    # Predict using the model
    prediction = model.predict(x)
    if(prediction[0]==0):
        all = "ไม่เป็น"
    elif(prediction[0]==1):
        all = "เป็น"
    


    # Return the result
    return jsonify({'prediction': all})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)