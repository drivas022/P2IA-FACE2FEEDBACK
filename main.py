import tkinter as tk
from tkinter import messagebox
import subprocess
from src.entrenamiento import entrenar_modelo
from src.evaluacion import evaluar_modelo
import os

class ModernApp:
    def __init__(self):
        self.ventana = tk.Tk()
        self.setup_window()
        self.create_widgets()
        
    def setup_window(self):
        self.ventana.title("Reconocimiento Facial para Feedback en Clases")
        self.ventana.geometry("500x700")
        self.ventana.configure(bg="#f8fafc")
        self.ventana.resizable(False, False)
        
        # Centrar ventana
        self.ventana.update_idletasks()
        x = (self.ventana.winfo_screenwidth() // 2) - (500 // 2)
        y = (self.ventana.winfo_screenheight() // 2) - (700 // 2)
        self.ventana.geometry(f"500x700+{x}+{y}")
        
    def create_widgets(self):
        # Header
        header_frame = tk.Frame(self.ventana, bg="#f8fafc", height=100)
        header_frame.pack(fill="x", padx=30, pady=(40, 20))
        header_frame.pack_propagate(False)
        
        # Título principal
        titulo = tk.Label(
            header_frame,
            text="🧠 AI Emotion Recognition",
            font=("SF Pro Display", 28, "bold"),
            bg="#f8fafc",
            fg="#1e293b"
        )
        titulo.pack(pady=(10, 5))
        
        # Subtítulo
        subtitulo = tk.Label(
            header_frame,
            text="Sistema de análisis emocional para clases",
            font=("SF Pro Display", 14),
            bg="#f8fafc",
            fg="#64748b"
        )
        subtitulo.pack()
        
        # Contenedor principal
        main_container = tk.Frame(self.ventana, bg="#ffffff", relief="flat")
        main_container.pack(fill="both", expand=True, padx=30, pady=20)
        
        # Agregar sombra visual
        shadow_frame = tk.Frame(main_container, bg="#e2e8f0", height=2)
        shadow_frame.pack(fill="x")
        
        content_frame = tk.Frame(main_container, bg="#ffffff")
        content_frame.pack(fill="both", expand=True, padx=40, pady=40)
        
        # Botones estilizados
        self.create_modern_button(
            content_frame,
            "🚀 Entrenar Modelo",
            "Entrenar el modelo de reconocimiento facial desde cero",
            "#3b82f6",
            "#2563eb",
            self.entrenar
        )
        
        self.create_modern_button(
            content_frame,
            "📊 Evaluar Modelo",
            "Evaluar el rendimiento del modelo entrenado",
            "#10b981",
            "#059669",
            self.evaluar
        )
        
        self.create_modern_button(
            content_frame,
            "🎥 Detección en Tiempo Real",
            "Iniciar análisis emocional en vivo",
            "#8b5cf6",
            "#7c3aed",
            self.iniciar_inferencia
        )
        
        # Separador
        separator = tk.Frame(content_frame, height=1, bg="#e2e8f0")
        separator.pack(fill="x", pady=30)
        
        # Botón de salida
        self.create_exit_button(content_frame)
        
        # Footer
        footer = tk.Label(
            self.ventana,
            text="© 2025 Facial Recognition System",
            font=("SF Pro Display", 10),
            bg="#f8fafc",
            fg="#94a3b8"
        )
        footer.pack(pady=(0, 20))
        
    def create_modern_button(self, parent, titulo, descripcion, color_normal, color_hover, comando):
        # Contenedor del botón
        btn_frame = tk.Frame(parent, bg="#ffffff")
        btn_frame.pack(fill="x", pady=15)
        
        # Botón principal
        btn = tk.Button(
            btn_frame,
            text=titulo,
            font=("SF Pro Display", 16, "bold"),
            bg=color_normal,
            fg="white",
            relief="flat",
            cursor="hand2",
            command=comando,
            height=2,
            activebackground=color_hover,
            activeforeground="white",
            bd=0
        )
        btn.pack(fill="x", ipady=15)
        
        # Descripción del botón
        desc = tk.Label(
            btn_frame,
            text=descripcion,
            font=("SF Pro Display", 11),
            bg="#ffffff",
            fg="#64748b"
        )
        desc.pack(pady=(8, 0))
        
        # Efectos hover
        def on_enter(e):
            btn.configure(bg=color_hover)
            
        def on_leave(e):
            btn.configure(bg=color_normal)
            
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        
    def create_exit_button(self, parent):
        exit_btn = tk.Button(
            parent,
            text="✕ Salir",
            font=("SF Pro Display", 14),
            bg="#ef4444",
            fg="white",
            relief="flat",
            cursor="hand2",
            command=self.ventana.quit,
            activebackground="#dc2626",
            activeforeground="white",
            bd=0
        )
        exit_btn.pack(fill="x", ipady=10)
        
        # Efectos hover para botón de salida
        def on_enter(e):
            exit_btn.configure(bg="#dc2626")
            
        def on_leave(e):
            exit_btn.configure(bg="#ef4444")
            
        exit_btn.bind("<Enter>", on_enter)
        exit_btn.bind("<Leave>", on_leave)
        
    def entrenar(self):
        dialog = self.create_modern_dialog(
            "Confirmar Entrenamiento",
            "¿Deseas entrenar el modelo desde cero?\nEsto puede tomar varios minutos.",
            "Entrenar",
            "Cancelar"
        )
        
        if dialog:
            try:
                entrenar_modelo()
                self.show_success_message("Modelo entrenado y guardado correctamente.")
            except Exception as e:
                self.show_error_message(f"Error durante el entrenamiento: {str(e)}")
                
    def evaluar(self):
        if not os.path.exists("models/modelo_entrenado.h5"):
            self.show_error_message("Primero debes entrenar el modelo.")
            return
        try:
            evaluar_modelo()
            self.show_success_message("Evaluación completada. Revisa la consola para los resultados.")
        except Exception as e:
            self.show_error_message(f"Error durante la evaluación: {str(e)}")
            
    def iniciar_inferencia(self):
        if not os.path.exists("models/modelo_entrenado.h5"):
            self.show_error_message("Primero debes entrenar el modelo.")
            return
        try:
            subprocess.Popen(["python", "src/interfaz_tkinter.py"])
        except Exception as e:
            self.show_error_message(f"Error al iniciar la inferencia: {str(e)}")
            
    def create_modern_dialog(self, titulo, mensaje, btn_si, btn_no):
        dialog = tk.Toplevel(self.ventana)
        dialog.title(titulo)
        dialog.geometry("400x200")
        dialog.configure(bg="#ffffff")
        dialog.resizable(False, False)
        dialog.transient(self.ventana)
        dialog.grab_set()
        
        # Centrar dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (400 // 2)
        y = (dialog.winfo_screenheight() // 2) - (200 // 2)
        dialog.geometry(f"400x200+{x}+{y}")
        
        # Contenido del dialog
        content = tk.Frame(dialog, bg="#ffffff")
        content.pack(fill="both", expand=True, padx=30, pady=30)
        
        # Mensaje
        msg = tk.Label(
            content,
            text=mensaje,
            font=("SF Pro Display", 13),
            bg="#ffffff",
            fg="#374151",
            justify="center",
            wraplength=340
        )
        msg.pack(pady=(0, 30))
        
        # Botones
        btn_frame = tk.Frame(content, bg="#ffffff")
        btn_frame.pack(fill="x")
        
        result = [False]
        
        def on_yes():
            result[0] = True
            dialog.destroy()
            
        def on_no():
            result[0] = False
            dialog.destroy()
            
        btn_yes = tk.Button(
            btn_frame,
            text=btn_si,
            font=("SF Pro Display", 12, "bold"),
            bg="#3b82f6",
            fg="white",
            relief="flat",
            cursor="hand2",
            command=on_yes,
            width=12,
            pady=10
        )
        btn_yes.pack(side="right", padx=(10, 0))
        
        btn_no = tk.Button(
            btn_frame,
            text=btn_no,
            font=("SF Pro Display", 12),
            bg="#6b7280",
            fg="white",
            relief="flat",
            cursor="hand2",
            command=on_no,
            width=12,
            pady=10
        )
        btn_no.pack(side="right")
        
        dialog.wait_window()
        return result[0]
        
    def show_success_message(self, mensaje):
        self.show_custom_message("✅ Éxito", mensaje, "#10b981")
        
    def show_error_message(self, mensaje):
        self.show_custom_message("❌ Error", mensaje, "#ef4444")
        
    def show_custom_message(self, titulo, mensaje, color):
        dialog = tk.Toplevel(self.ventana)
        dialog.title(titulo)
        dialog.geometry("350x150")
        dialog.configure(bg="#ffffff")
        dialog.resizable(False, False)
        dialog.transient(self.ventana)
        dialog.grab_set()
        
        # Centrar dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (350 // 2)
        y = (dialog.winfo_screenheight() // 2) - (150 // 2)
        dialog.geometry(f"350x150+{x}+{y}")
        
        # Contenido
        content = tk.Frame(dialog, bg="#ffffff")
        content.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Título con color
        title_label = tk.Label(
            content,
            text=titulo,
            font=("SF Pro Display", 14, "bold"),
            bg="#ffffff",
            fg=color
        )
        title_label.pack(pady=(0, 10))
        
        # Mensaje
        msg = tk.Label(
            content,
            text=mensaje,
            font=("SF Pro Display", 11),
            bg="#ffffff",
            fg="#374151",
            justify="center",
            wraplength=310
        )
        msg.pack(pady=(0, 20))
        
        # Botón OK
        ok_btn = tk.Button(
            content,
            text="OK",
            font=("SF Pro Display", 12),
            bg=color,
            fg="white",
            relief="flat",
            cursor="hand2",
            command=dialog.destroy,
            width=10,
            pady=8
        )
        ok_btn.pack()
        
    def run(self):
        self.ventana.mainloop()

if __name__ == "__main__":
    app = ModernApp()
    app.run()