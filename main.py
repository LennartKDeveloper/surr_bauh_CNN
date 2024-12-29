from src.train import train_model
from src.predict import predict_image

# Datenverzeichnis und Modellpfad
DATA_DIR = "./dataset"
MODEL_PATH = "./saved_model/lennet.keras"
image_path_1 = "./usage/surr_use_2.jpg"
image_path_2 = "./usage/bauh_use_1.jpg"




if __name__ == "__main__":
  # Training des Modells
  # train_model(DATA_DIR, MODEL_PATH, epochs=10)

  # Testen mit einem neuen Bild
  arr = predict_image(MODEL_PATH, image_path_2)

  print(f"The predicted class is: {arr[0]}, with an accurracy of {arr[1]*100:.2f} %")
