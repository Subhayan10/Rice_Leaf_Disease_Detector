import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

IMG_SIZE = 224  # same as training

def preprocess_image(img_path):
    """Load and preprocess image for prediction."""
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize
    return img_array
