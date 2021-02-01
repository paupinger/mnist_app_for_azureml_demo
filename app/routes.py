import numpy as np
from app import app
from flask import render_template, request, jsonify
from app.predictors import predict_by_endpoint


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # normalized
    input_data = (255 - np.array(request.json))

    result_of_endpoint = predict_by_endpoint(input_data)
    return jsonify(data=[result_of_endpoint])
