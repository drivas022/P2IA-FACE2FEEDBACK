import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def entrenar_modelo():
    # Aumentar imágenes del dataset con data augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        'data/train',
        target_size=(48, 48),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        'data/train',
        target_size=(48, 48),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=32,
        shuffle=False,
        subset='validation'
    )

    model = Sequential()

    # Primer bloque
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)))
    model.add(MaxPooling2D(2,2))

    # Segundo bloque
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))

    # Tercer bloque
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.25))

    # Clasificador
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    os.makedirs("models", exist_ok=True)
    checkpoint = ModelCheckpoint("models/modelo_entrenado.h5", save_best_only=True, monitor="val_loss", mode="min")
    earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=30,
        callbacks=[checkpoint, earlystop]
    )

    print("✅ Modelo entrenado y guardado en models/modelo_entrenado.h5")
