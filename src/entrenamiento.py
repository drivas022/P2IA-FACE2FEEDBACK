import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 30

def cargar_datos():
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        'data/train',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        'data/train',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_generator, val_generator

def construir_modelo():
    modelo = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return modelo

def entrenar_modelo():
    train_gen, val_gen = cargar_datos()
    modelo = construir_modelo()
    early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    modelo.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, callbacks=[early])
    modelo.save('models/modelo_entrenado.h5')
    print("✅ Modelo guardado en models/modelo_entrenado.h5")

if __name__ == '__main__':
    entrenar_modelo()
