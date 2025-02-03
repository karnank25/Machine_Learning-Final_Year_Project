import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
#%matplotlib inline 

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
import random
from cv2 import resize
import cv2
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import normalize
from glob import glob
def process(path):
    classes = ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]
    model_path = './model_v1_ALL.h5'

    try:
        gru_model = load_model(model_path)
    except Exception as e:
        print("Error loading the model:", e)
        return None
    
    try:
        frame = cv2.imread(path)
        if frame is None:
            print("Error: Unable to load the image from", path)
            return None
        img = cv2.resize(frame, (224, 224))
    except Exception as e:
        print("Error reading or resizing the image:", e)
        return None
    
    try:
        img = img.astype('float32') / 255.0  
        img = np.expand_dims(img, axis=0)
        img = img.reshape(1, 224, 224, 3)
    except Exception as e:
        print("Error reshaping the image:", e)
        return None
    
    try:
        predictions = gru_model.predict(img)
        prediction_index = np.argmax(predictions)
        prediction = classes[prediction_index]
        return prediction
    except Exception as e:
        print("Error making predictions:", e)
        return None