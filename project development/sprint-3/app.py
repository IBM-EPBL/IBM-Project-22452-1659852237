import requests
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session
from waitress import serve


app = Flask(__name__)

#Loading the models
vmodel = load_model("vegetable.h5")
fmodel = load_model("fruit.h5")

#Home page
@app.route('/')
def home():
    return render_template('index.html')

#Prediction page
@app.route('/prediction')
def prediction():
    return render_template('plants.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img = image.load_img(file_path, target_size=(128, 128))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        plant=request.form['plants']
        print(plant)
        if(plant=="fruit"):
            preds = fmodel.predict(x)
            print(preds[0])
            df=pd.read_excel("fruit.xlsx")
            print(df.iloc[preds[0]]['CAUTION'])
        else:
            preds = vmodel.predict(x)
            print(preds[0])
            df=pd.read_excel("VEGETABLE.xlsx")
            print(df.iloc[preds[0]]['CAUTION'])
        return render_template("results.html",data=df.iloc[preds[0]]['CAUTION'])

if __name__=="__main__":
    app.run(debug=True)
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)