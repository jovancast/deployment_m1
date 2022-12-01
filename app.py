from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

# App Initialization
app = Flask(__name__)

# Load the Model
pipeline = pickle.load(open('pipeline.pkl','rb'))
model = tf.keras.models.load_model('model.h5')

# Endpoint for Homepage
@app.route("/")
def home():
    return "<h1>It Works!</h1>"

# Endpoint for Prediction
@app.route("/predict", methods=['POST'])
def model_predict():
    args = request.json
    new_data = {'gender': args.get('gender'),
    'SeniorCitizen': args.get('SeniorCitizen'),
    'Partner': args.get('Partner'),
    'Dependents': args.get('Dependents'),
    'tenure': args.get('tenure'),
    'PhoneService': args.get('PhoneService'),
    'MultipleLines': args.get('MultipleLines'),
    'InternetService': args.get('InternetService'),
    'OnlineSecurity': args.get('OnlineSecurity'),
    'OnlineBackup': args.get('OnlineBackup'),
    'DeviceProtection': args.get('DeviceProtection'),
    'TechSupport': args.get('TechSupport'),
    'StreamingTV': args.get('StreamingTV'),
    'StreamingMovies': args.get('StreamingMovies'),
    'Contract': args.get('Contract'),
    'PaperlessBilling': args.get('PaperlessBilling'),
    'PaymentMethod': args.get('PaymentMethod'),
    'MonthlyCharges': args.get('MonthlyCharges'),
    'TotalCharges': args.get('TotalCharges')}

    new_data = pd.DataFrame([new_data])
    print('New Data : ', new_data)

    #pipeline
    X = pipeline.transform(new_data)

    # Predict
    y_label = ['No','Yes']
    y_pred = int(np.round(model.predict(X)[0][0]))

    # Return the Response
    response = jsonify(
      result = str(y_pred), 
      label_names = y_label[y_pred])

    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)