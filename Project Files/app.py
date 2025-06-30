from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load the trained model
model = load_model("model.h5")

# Set the directory path to your training folder to extract class names
train_dir = "C:/Users/ashar/Downloads/enchantedwings/train"
class_labels = sorted(os.listdir(train_dir))  # This gives all 75 species names in order

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        file = request.files["image"]
        filename = file.filename
        filepath = os.path.join("static", filename)
        file.save(filepath)

        # Preprocess image
        img = load_img(filepath, target_size=(128, 128))  # Same size as used in training
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_index]  # Get species name

        return render_template("predict.html", prediction=predicted_class, image_path=filepath)

if __name__ == "__main__":
    app.run(debug=True)
