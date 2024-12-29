import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def predict_image(model_path, image_path, img_size=(150, 150)):
    # Modell laden
    model = tf.keras.models.load_model(model_path)
    
    # Bild vorbereiten
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Batch-Dimension
    
    # Vorhersage
    prediction = model.predict(img_array)
    return ["Surrealism" if prediction[0][0] > 0.5 else "Bauhaus", (prediction[0][0]) if prediction[0][0] > 0.5 else (1-prediction[0][0])]
