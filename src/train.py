import os
from src.data_loader import load_data
from src.model import create_model



def train_model(data_dir, save_path, epochs=10):
    # Daten laden
    train_dataset, val_dataset = load_data(data_dir)
    
    # Modell erstellen
    model = create_model()
    
    # Training
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs
    )
    
    # Modell speichern
    os.makedirs("./saved_model", exist_ok=True)
    model.save(save_path)
    print(f"Model saved at {save_path}")
