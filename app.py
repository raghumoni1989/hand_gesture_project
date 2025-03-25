import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
from flask import Flask, render_template, url_for, request
import numpy as np
from tensorflow.keras.models import load_model
import sqlite3
import shutil
import pygame
import time
from gtts import gTTS
from mutagen.mp3 import MP3
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

@app.route('/')
def index():
    print("[INFO] Rendering index page")
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':
        print("[INFO] User login attempt")
        
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        print(f"[DEBUG] Received login credentials: {name}, {password}")

        query = "SELECT name, password FROM user WHERE name = ? AND password = ?"
        cursor.execute(query, (name, password))
        result = cursor.fetchall()

        if len(result) == 0:
            print("[WARNING] Incorrect login credentials")
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided, Try Again')
        else:
            print("[INFO] User logged in successfully")
            return render_template('userlog.html')

    return render_template('index.html')

@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':
        print("[INFO] User registration attempt")
        
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()
        
        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        print(f"[DEBUG] Registration data: {name}, {mobile}, {email}, {password}")
        
        command = "CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"
        cursor.execute(command)
        
        cursor.execute("INSERT INTO user VALUES (?, ?, ?, ?)", (name, password, mobile, email))
        connection.commit()
        
        print("[INFO] User successfully registered")
        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        print("[INFO] Processing image upload")
        
        dirPath = "static/images"
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
            print("[DEBUG] Created directory: static/images")
        
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(os.path.join(dirPath, fileName))
            print(f"[DEBUG] Deleted old image: {fileName}")
        
        fileName = request.form['filename']
        dst = "static/images"
        shutil.copy("test/" + fileName, dst)
        print(f"[INFO] Copied file {fileName} to {dst}")
        
        model = load_model('gesture_classifier.h5')
        print("[INFO] Model loaded successfully")
        
        with open('class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
            print("[INFO] Class names loaded")

        def preprocess_input_image(path):
            img = load_img(path, target_size=(150,150))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize the image
            return img_array

        def predict_single_image(path):
            input_image = preprocess_input_image(path)
            prediction = model.predict(input_image)
            print(f"[DEBUG] Model prediction output: {prediction}")
            predicted_class_index = np.argmax(prediction)
            predicted_class = class_names[predicted_class_index]
            confidence = prediction[0][predicted_class_index]
            return predicted_class, confidence 

        predicted_class, confidence = predict_single_image(os.path.join(dst, fileName))
        print(f"[INFO] Predicted Class: {predicted_class} with confidence {confidence:.2%}")
        
        myobj = gTTS(text=predicted_class, lang='en', slow=False)
        myobj.save("voice.mp3")
        
        song = MP3("voice.mp3")
        pygame.mixer.init()
        pygame.mixer.music.load('voice.mp3')
        pygame.mixer.music.play()
        time.sleep(song.info.length)
        pygame.quit()
        print("[INFO] Audio playback complete")
        
        accuracy = f"The predicted image is {predicted_class} with a confidence of {confidence:.2%}"
        return render_template('results.html', status=predicted_class, accuracy=accuracy, ImageDisplay=f"http://127.0.0.1:5000/static/images/{fileName}")
    
    return render_template('userlog.html')

@app.route('/logout')
def logout():
    print("[INFO] User logged out")
    return render_template('index.html')

if __name__ == "__main__":
    print("[INFO] Starting Flask server")
    app.run(debug=True, use_reloader=False)