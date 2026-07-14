"""
EPIC 5: Application Building
Run AFTER step7_evaluate_save.py
Run: python app.py
Open: http://127.0.0.1:5000
"""
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model  = pickle.load(open('models/model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/findyourcrop')
def findyourcrop():
    return render_template('findyourcrop.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    features = np.array([int_features])
    features = scaler.transform(features)
    prediction = model.predict(features)
    output = prediction[0]
    return render_template('findyourcrop.html',
                           prediction_text='Best crop for given conditions is {}'.format(output))


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
