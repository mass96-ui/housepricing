import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# LOAD MODEL + SCALER (BOTH REQUIRED)
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():

    data = request.json['data']

    input_data = np.array(list(data.values())).reshape(1, -1)

    scaled_data = scaler.transform(input_data)

    prediction = regmodel.predict(scaled_data)

    return jsonify(float(prediction[0]))


if __name__ == "__main__":
    app.run(debug=True)