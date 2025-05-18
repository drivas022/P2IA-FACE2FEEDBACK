import tkinter as tk
from tkinter import messagebox
import cv2
import threading
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from collections import Counter
import os
from datetime import datetime

modelo = load_model('models/modelo_entrenado.h5')
emociones = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

recomendaciones = {
    'angry': 'Sugerencia: Tomar una pausa.',
    'disgust': 'Sugerencia: Cambiar de actividad.',
    'fear': 'Sugerencia: Repetir el contenido.',
    'happy': 'Sugerencia: Continuar con la clase.',
    'neutral': 'Sugerencia: Seguir así.',
    'sad': 'Sugerencia: Reforzar explicación.',
    'surprise': 'Sugerencia: Hacer dinámica interactiva.'
}

class EmotionApp:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("Reconocimiento Facial para Feedback en Clases")
        self.cap = None
        self.running = False
        self.emociones_detectadas = []

        controles = tk.Frame(ventana)
        controles.pack()

        self.iniciar_btn = tk.Button(controles, text="Iniciar", command=self.iniciar)
        self.iniciar_btn.grid(row=0, column=0, padx=5)

        self.finalizar_btn = tk.Button(controles, text="Finalizar", command=self.finalizar)
        self.finalizar_btn.grid(row=0, column=1, padx=5)

        self.lienzo = tk.Label(ventana)
        self.lienzo.pack()

        self.recomendacion_texto = tk.StringVar()
        self.recomendacion_texto.set("Recomendación: ---")
        self.recomendacion_label = tk.Label(ventana, textvariable=self.recomendacion_texto, font=("Arial", 14), fg="blue")
        self.recomendacion_label.pack(pady=10)

    def predecir_emocion(self, rostro):
        rostro = cv2.resize(rostro, (48, 48))
        rostro = rostro.astype('float32') / 255.0
        rostro = np.expand_dims(rostro, axis=0)
        rostro = np.expand_dims(rostro, axis=-1)
        pred = modelo.predict(rostro)
        emocion = emociones[np.argmax(pred)]
        return emocion

    def actualizar_camara(self):
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rostros = detector.detectMultiScale(gris, 1.3, 5)

            for (x, y, w, h) in rostros:
                rostro = gris[y:y+h, x:x+w]
                emocion = self.predecir_emocion(rostro)
                self.emociones_detectadas.append(emocion)
                recomendacion = recomendaciones.get(emocion, "Recomendación: ---")
                self.recomendacion_texto.set(recomendacion)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, emocion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.lienzo.imgtk = imgtk
            self.lienzo.configure(image=imgtk)
            self.ventana.update()

        if self.cap:
            self.cap.release()
        self.lienzo.config(image='')

    def iniciar(self):
        self.emociones_detectadas.clear()
        self.cap = cv2.VideoCapture(0)
        self.running = True
        threading.Thread(target=self.actualizar_camara).start()

    def finalizar(self):
        self.running = False
        if self.cap:
            self.cap.release()

        conteo = Counter(self.emociones_detectadas)
        if not conteo:
            messagebox.showinfo("Resumen", "No se detectaron emociones.")
            return

        resumen = "\n".join([f"{emo}: {conteo[emo]}" for emo in conteo])
        emocion_principal = conteo.most_common(1)[0][0]
        recomendacion = recomendaciones.get(emocion_principal, "No se pudo generar recomendación.")

        # Crear carpeta results y guardar .txt
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta = f"results/detalle_sesion_{timestamp}.txt"

        with open(ruta, "w", encoding="utf-8") as f:
            f.write("🧠 RESULTADO DE LA SESIÓN\n")
            f.write(f"Fecha y hora: {datetime.now()}\n\n")
            f.write("Emociones detectadas:\n")
            for emo, count in conteo.items():
                f.write(f"- {emo}: {count}\n")
            f.write(f"\nEmoción predominante: {emocion_principal}\n")
            f.write(f"Recomendación final: {recomendacion}\n")

        messagebox.showinfo("Resumen del análisis",
                            f"Emociones detectadas:\n{resumen}\n\nRecomendación final:\n{recomendacion}\n\nResultado guardado en:\n{ruta}")

        self.recomendacion_texto.set("Recomendación: ---")

if __name__ == "__main__":
    ventana = tk.Tk()
    app = EmotionApp(ventana)
    ventana.mainloop()
