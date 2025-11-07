from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
from utils.preprocess import preprocess_image

# Initialize Flask app
app = Flask(__name__)

# Load trained model
MODEL_PATH = "model/rice_resnet50_finetuned.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Load class labels (must match your training classes)
CLASS_NAMES = ['Bacterial leaf blight', 'Brown spot', 'Healthy' , 'Leaf Blast' , 'Leaf Scald' , 'Narrow Brown Spot']  # 👈 change based on your dataset

@app.route('/')
def home():
    return "✅ Rice Disease Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    # Preprocess image
    img_array = preprocess_image(file_path)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    # Clean up
    os.remove(file_path)

    return jsonify({
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4)
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
