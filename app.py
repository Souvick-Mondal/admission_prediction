import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
 
app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    #For rendering results on HTML GUI
    
    int_features=[float(x)for x in request.form.values()]
    final_features=[np.array(int_features)]
    
    pred=model.predict(final_features)
    output = round(pred[0], 2)
    result=output*100
    return render_template('home.html', preditction_text='Percentage should be $ {}%'.format(result))




if __name__ == "__main__":
    app.run(debug=True)
