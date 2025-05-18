import tkinter as tk
from tkinter import messagebox
import subprocess
from src.entrenamiento import entrenar_modelo
from src.evaluacion import evaluar_modelo
import os

def entrenar():
    respuesta = messagebox.askyesno("Confirmar", "¿Deseas entrenar el modelo desde cero?")
    if respuesta:
        entrenar_modelo()
        messagebox.showinfo("Éxito", "Modelo entrenado y guardado correctamente.")

def evaluar():
    if not os.path.exists("models/modelo_entrenado.h5"):
        messagebox.showerror("Error", "Primero debes entrenar el modelo.")
        return
    evaluar_modelo()

def iniciar_inferencia():
    if not os.path.exists("models/modelo_entrenado.h5"):
        messagebox.showerror("Error", "Primero debes entrenar el modelo.")
        return
    subprocess.Popen(["python", "src/interfaz_tkinter.py"])

# Crear ventana principal
ventana = tk.Tk()
ventana.title("Reconocimiento Facial para Feedback en Clases")
ventana.geometry("400x250")

# Título
titulo = tk.Label(ventana, text="Proyecto de Reconocimiento Facial", font=("Helvetica", 14, "bold"))
titulo.pack(pady=20)

# Botones
btn_entrenar = tk.Button(ventana, text="Entrenar Modelo", width=25, height=2, command=entrenar)
btn_entrenar.pack(pady=5)

btn_evaluar = tk.Button(ventana, text="Evaluar Modelo", width=25, height=2, command=evaluar)
btn_evaluar.pack(pady=5)

btn_iniciar = tk.Button(ventana, text="Iniciar Detección en Tiempo Real", width=25, height=2, command=iniciar_inferencia)
btn_iniciar.pack(pady=5)

# Cerrar
btn_salir = tk.Button(ventana, text="Salir", width=25, height=2, command=ventana.quit)
btn_salir.pack(pady=10)

ventana.mainloop()
