import tensorflow as tf

def load_data(data_dir, img_size=(150, 150), batch_size=32):
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir + '/train',
        image_size=img_size,
        batch_size=batch_size
    )
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir + '/validation',
        image_size=img_size,
        batch_size=batch_size
    )
    # Optimierung
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    return train_dataset, val_dataset
