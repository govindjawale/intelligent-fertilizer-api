from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# 1. Load the model (Loads once when the server boots up)
print("Loading Model...")
model = tf.keras.models.load_model('scopus_plant_disease_model.keras')

# 2. The 38 Class Names
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# 3. Create the API Endpoint
@app.route('/predict', methods=['POST'])
def predict_disease():
    if 'file' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    # Read the image sent via HTTPS
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).resize((224, 224))
    
    # Convert to array for the AI
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    
    # Make Prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]) * 100)
    disease_name = class_names[predicted_index]

    # 4. Hardware Logic (The Smart Thresholds)
    pump_1 = 0
    pump_2 = 0
    
    if confidence >= 65.0:
        if "healthy" in disease_name.lower():
            pump_2 = 1 # Give fertilizer/water
        else:
            pump_1 = 1 # Spray pesticide
            
    # Return the data as JSON to the ESP32
    return jsonify({
        "disease": disease_name,
        "confidence": round(confidence, 2),
        "pump_1_pesticide": pump_1,
        "pump_2_fertilizer": pump_2
    })

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(host='0.0.0.0', port=5000)