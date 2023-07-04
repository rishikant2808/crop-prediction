# import os
# import sys

# import pandas as pd

# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import SGD, Adam
# from keras.utils.np_utils import to_categorical
# Flask
from flask import Flask, url_for, request, render_template


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model

# Some utilites
import numpy as np


# from sklearn.metrics import classification_report, confusion_matrix


# Declare a flask app
app = Flask(__name__)

# model=pickle.load(open('model.pkl','rb'))
print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
MODEL_PATH = 'model/my_model.h5'
model=load_model(MODEL_PATH)
model.make_predict_function()         # Necessary
print('Model loaded. Start serving...')

# def model_predict(data, model):
  
@app.route('/')
def hello_world():
    return render_template("index.html");
@app.route('/predictor')
def hello():
    return render_template("predictor.html");
@app.route('/contact')
def hellos():
    return render_template("contact.html");



# @app.route('/supervisor')
# def supervisor():
#     return render_template("projectsupervisor.html");

# @app.route('/datahandlingteam')
# def datahandlingteam():
#     return render_template("datahandlingteam.html");
# @app.route('/modeltrainers')
# def modeltrainers():
#     return render_template("modeltraining.html");
# @app.route('/appdevelopers')
# def appdev():
#     return render_template("appdeveloper.html");
# @app.route('/webdevelopers')
# def webdev():
#     return render_template("webdev.html");
# @app.route('/')
# def hello_world():
#     return render_template("index.html");

# def model_predict(img, model):
#     img = img.resize((224, 224))

#     # Preprocessing the image
#     x = image.img_to_array(img)
#     # x = np.true_divide(x, 255)
#     x = np.expand_dims(x, axis=0)

#     # Be careful how your trained model deals with the input
#     # otherwise, it won't make correct prediction!
#     x = preprocess_input(x, mode='tf')

#     preds = model.predict(x)
#     return preds

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == "POST":
        inputvalues=[float(x) for x in request.form.values()]
        inputvalues = np.array(inputvalues)
        inputvalues = inputvalues.reshape(1, -1)
        print(inputvalues)
        # prediction = model.predict(inputvalues)
        ans=''
        prediction = np.argmax(model.predict(inputvalues), axis=1)
        if prediction == [0]:
            ans='Cauliflower'
        elif prediction == [1]:
            ans='Onion'
        elif prediction == [2]:
            ans='Ginger'
        elif prediction == [3]:
            ans='Garlic'
        else:
            ans='Tomato'       

    # if output>str(0.5):
    #     return render_template('forest_fire.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),bhai="kuch karna hain iska ab?")
    # else:
    #     return render_template('forest_fire.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),bhai="Your Forest is Safe for now")
    return render_template('predictor.html',pred=ans)

if __name__ == '__main__':
    app.run(debug=True)