# src/inferencia_en_tiempo_real.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model

modelo = load_model('models/modelo_entrenado.h5')
emociones = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def predecir_emocion(frame):
    rostro = cv2.resize(frame, (48, 48))
    rostro = rostro.astype('float32') / 255.0
    rostro = np.expand_dims(rostro, axis=0)
    rostro = np.expand_dims(rostro, axis=-1)
    pred = modelo.predict(rostro)
    return emociones[np.argmax(pred)]

def iniciar_camara():
    cap = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostros = detector.detectMultiScale(gris, 1.3, 5)

        for (x, y, w, h) in rostros:
            rostro = gris[y:y+h, x:x+w]
            emocion = predecir_emocion(rostro)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emocion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        cv2.imshow('Reconocimiento de emociones', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    iniciar_camara()
