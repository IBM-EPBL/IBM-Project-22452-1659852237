import requests
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template, Markup, redirect, url_for
from disease import disease_dic
import os
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session

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

#Feedback Page
@app.route('/feedback')
def feedback():
    return render_template('contact.html')

#About Page
@app.route('/about')
def about():
    return render_template('about.html')

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
            preds = np.argmax(fmodel.predict(x),axis=1)
            print(preds[0])
            fruit_index = ['Apple___Black_rot', 'Apple___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',  'Peach___Bacterial_spot',  'Peach___healthy']
            pred = fruit_index[preds[0]]
        else:
            preds = np.argmax(vmodel.predict(x),axis=1)
            print(preds[0])
            veg_index = ['Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',  'Potato___healthy',  'Tomato___Bacterial_spot', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot']
            pred = veg_index[preds[0]]
        prediction = Markup(str(disease_dic[pred]))
        return render_template("results.html", disease=prediction, img=img)

if __name__=="__main__":
    app.run(debug=False)
