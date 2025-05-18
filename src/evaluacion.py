from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

IMG_SIZE = (48, 48)
BATCH_SIZE = 64

def evaluar_modelo():
    modelo = load_model('models/modelo_entrenado.h5')

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False
    )

    predicciones = modelo.predict(test_generator)
    etiquetas_pred = np.argmax(predicciones, axis=1)
    etiquetas_verdaderas = test_generator.classes
    etiquetas_nombres = list(test_generator.class_indices.keys())

    print(classification_report(etiquetas_verdaderas, etiquetas_pred, target_names=etiquetas_nombres))
    print("Matriz de confusión:")
    print(confusion_matrix(etiquetas_verdaderas, etiquetas_pred))

if __name__ == '__main__':
    evaluar_modelo()
