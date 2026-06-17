import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# -------------------------------
# SAFE PATH FIX (IMPORTANT)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

regmodel = pickle.load(open(os.path.join(BASE_DIR, 'regmodel.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb'))


# -------------------------------
# HOME PAGE
# -------------------------------
@app.route('/')
def home():
    return render_template('home.html')


# -------------------------------
# API PREDICTION (JSON INPUT)
# -------------------------------
@app.route('/predict_api', methods=['POST'])
def predict_api():

    data = request.json['data']

    input_data = np.array(list(data.values())).reshape(1, -1)

    scaled_data = scaler.transform(input_data)

    prediction = regmodel.predict(scaled_data)

    return jsonify(float(prediction[0]))


# -------------------------------
# HTML FORM PREDICTION
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():

    data = [float(x) for x in request.form.values()]

    final_input = scaler.transform(np.array(data).reshape(1, -1))

    output = regmodel.predict(final_input)[0]

    return render_template(
        "home.html",
        prediction_text=f"The house price prediction is {output}"
    )


# -------------------------------
# RUN APP
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)