import tensorflow as tf
import numpy as np
import os

# 1. The 38 class names
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

print("Waking up the Neural Network...")
# 2. Load model ONLY ONCE outside the loop for maximum performance
model = tf.keras.models.load_model('scopus_plant_disease_model.keras')

# 3. Point the script to your new image folder
image_folder = 'test_images'

# Check if the folder exists to prevent crashes
if not os.path.exists(image_folder):
    print(f"Error: Could not find the '{image_folder}' directory. Please create it.")
else:
    print(f"Scanning '{image_folder}' for images...\n")
    
    # 4. Loop through every single file inside the folder
    for filename in os.listdir(image_folder):
        
        # Only process actual image files
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            
            # Load and format the image
            img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) 
            
            # Make the prediction (verbose=0 keeps terminal clean)
            predictions = model.predict(img_array, verbose=0)
            predicted_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            
            # 5. Output the results with a Smart Multi-Tiered Threshold
            print("="*45)
            print(f"FILE: {filename}")
            
            # TIER 1: High Confidence (Lab Quality)
            if confidence >= 85.0:
                print(f"DISEASE DETECTED: {class_names[predicted_index]}")
                print(f"AI CONFIDENCE:    {confidence:.2f}% (High)")
                print("HARDWARE SIGNAL:  [1] - TRIGGER PUMP (STANDARD DOSE)")
            
            # TIER 2: Medium Confidence (Real-World/Noisy Background)
            elif confidence >= 65.0:
                print(f"DISEASE DETECTED: {class_names[predicted_index]}")
                print(f"AI CONFIDENCE:    {confidence:.2f}% (Medium - Real World)")
                print("HARDWARE SIGNAL:  [1] - TRIGGER PUMP (STANDARD DOSE)")
                
            # TIER 3: Low Confidence (Uncertain/Different Disease)
            else:
                print(f"BEST GUESS:       {class_names[predicted_index]}")
                print(f"AI CONFIDENCE:    {confidence:.2f}% (Too Low)")
                print("HARDWARE SIGNAL:  [0] - DO NOT SPRAY")
            
            print("="*45 + "\n")