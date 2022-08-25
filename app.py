# Serve model as a flask application

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

model = None
app = Flask(__name__)


def load_model():
    global model
    # model variable refers to the global variable
    with open('iris_trained_model.pkl', 'rb') as f:
        model = pickle.load(f)


@app.route('/')
def home_endpoint():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Works only for a single sample
    if request.method == 'POST':
        int_features = [int(x) for x in request.form.values()]
        # print(request.form.values())
        final_features = np.array(int_features)[np.newaxis, :]
        prediction = model.predict(final_features)

        output = round(prediction[0],2)

        return render_template('index.html', prediction_text='Iris should be type {}'.format(output))

        # data = request.get_json()  # Get data posted as a json
        # data = np.array(data)[np.newaxis, :]  # converts shape from (4,) to (1, 4)
        # prediction = model.predict(data)  # runs globally loaded model on the data
    # return str(prediction[0])


if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run(debug=True)
