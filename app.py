
from flask import Flask, render_template, request
import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import os

app = Flask(__name__)

# Load models
symptom_model = pickle.load(open("symptom_model.pkl", "rb"))
cnn_model = load_model("models/cnn_model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/symptom', methods=['GET', 'POST'])
def symptom():
    if request.method == 'POST':
        symptoms = request.form['symptoms'].split(',')
        all_symptoms = ['fever', 'cough', 'headache', 'fatigue']
        input_vector = [1 if s.strip().lower() in symptoms else 0 for s in all_symptoms]
        prediction = symptom_model.predict([input_vector])
        return render_template('result.html', result=prediction[0])
    return render_template('symptom.html')

@app.route('/image', methods=['POST'])
def image_diagnosis():
    file = request.files['image']
    filepath = os.path.join('static', file.filename)
    file.save(filepath)
    
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = cnn_model.predict(img_array)
    result = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
