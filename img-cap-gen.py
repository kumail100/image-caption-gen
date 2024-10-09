import os
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.preprocessing import image as keras_image

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained InceptionV3 model
inception_model = inception_v3.InceptionV3(weights='imagenet')

# Helper function to preprocess the image
def preprocess_image(img_path):
    img = keras_image.load_img(img_path, target_size=(299, 299))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return inception_v3.preprocess_input(img_array)

# Simple caption generator based on predicted class names
def generate_caption(img_path):
    preprocessed_image = preprocess_image(img_path)
    predictions = inception_model.predict(preprocessed_image)
    decoded_predictions = inception_v3.decode_predictions(predictions, top=3)[0]
    caption = ", ".join([item[1] for item in decoded_predictions])
    return f"AI-generated caption: {caption}"

# API route for image upload
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)

    # Generate caption
    caption = generate_caption(img_path)

    return jsonify({"caption": caption})

# Run the app
if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='0.0.0.0', port=5000)
