from flask import render_template, request, jsonify,Flask
import flask
import numpy as np
import traceback
import pickle
import pandas as pd
 
 
# App definition
app = Flask(__name__,template_folder='templates')
 
# importing models
model = pickle.load(open('model1.pkl', 'rb'))
 
@app.route('/')
def home():
    return render_template('DOCTYPE.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('DOCTYPE.html', prediction_text='Motortemperature should be  {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([(list(data.values()))])

    output = prediction[2]
    return jsonify(output)
if __name__ == "__main__":
   app.run()    