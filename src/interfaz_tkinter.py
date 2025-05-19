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
from correo_utils import enviar_recomendacion_por_correo

# Configuración matplotlib (debe ir antes que la importación)
import matplotlib
matplotlib.use('TkAgg')  # Usar backend TkAgg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Cargar modelo entrenado
modelo = load_model('models/modelo_entrenado.h5')
emociones = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Paleta de colores moderna para cada emoción
emotion_colors = {
    'angry': '#ef4444',
    'disgust': '#8b5cf6',
    'fear': '#f59e0b',
    'happy': '#10b981',
    'neutral': '#6b7280',
    'sad': '#3b82f6',
    'surprise': '#ec4899'
}

# Recomendaciones mejoradas para cada emoción
recomendaciones = {
    'angry': 'Considera una pausa para relajar el ambiente',
    'disgust': 'Cambia la actividad o enfoque del tema',
    'fear': 'Refuerza la explicación paso a paso',
    'happy': 'Continúa con este ritmo de clase',
    'neutral': 'Mantén el enfoque actual',
    'sad': 'Motiva y engage más a los estudiantes',
    'surprise': 'Aprovecha para hacer actividades dinámicas'
}

class ModernEmotionApp:
    def __init__(self, ventana):
        self.ventana = ventana
        self.setup_window()
        self.cap = None
        self.running = False
        self.emociones_detectadas = []
        self.frame_count = 0
        self.start_time = None
        self.create_widgets()
        
    def setup_window(self):
        """Configurar ventana principal"""
        self.ventana.title("🧠 Análisis Emocional en Tiempo Real")
        self.ventana.configure(bg="#f8fafc")
        # Cambiar maximizar por geometry fija para evitar problemas
        self.ventana.geometry("1400x900")
        
    def create_widgets(self):
        """Crear todos los widgets de la interfaz"""
        try:
            # Header principal
            header_frame = tk.Frame(self.ventana, bg="#1e293b", height=80)
            header_frame.pack(fill="x")
            header_frame.pack_propagate(False)
            
            # Título en header
            title = tk.Label(
                header_frame,
                text="🎥 Análisis Emocional en Tiempo Real",
                font=("Arial", 20, "bold"),  # Cambiar font por compatibilidad
                bg="#1e293b",
                fg="white"
            )
            title.pack(pady=20)
            
            # Contenedor principal
            main_container = tk.Frame(self.ventana, bg="#f8fafc")
            main_container.pack(fill="both", expand=True, padx=20, pady=20)
            
            # Panel izquierdo - Video
            left_panel = tk.Frame(main_container, bg="#ffffff", relief="flat")
            left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
            
            # Frame para el video
            video_frame = tk.Frame(left_panel, bg="#ffffff")
            video_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            self.video_label = tk.Label(
                video_frame,
                text="📹 Video en vivo aparecerá aquí",
                font=("Arial", 14),
                bg="#f1f5f9",
                fg="#64748b",
                relief="flat",
                bd=2
            )
            self.video_label.pack(fill="both", expand=True)
            
            # Controles de video
            controls_frame = tk.Frame(left_panel, bg="#ffffff")
            controls_frame.pack(fill="x", padx=20, pady=20)
            
            self.start_btn = self.create_control_button(
                controls_frame, "▶ Iniciar", "#10b981", "#059669", self.iniciar
            )
            self.start_btn.pack(side="left", padx=(0, 10))
            
            self.stop_btn = self.create_control_button(
                controls_frame, "⏹ Detener", "#ef4444", "#dc2626", self.finalizar
            )
            self.stop_btn.pack(side="left", padx=(0, 10))
            self.stop_btn.configure(state="disabled")
            
            self.save_btn = self.create_control_button(
                controls_frame, "💾 Guardar", "#3b82f6", "#2563eb", self.guardar_sesion
            )
            self.save_btn.pack(side="left", padx=(0, 10))
            
            # Panel derecho - Análisis
            right_panel = tk.Frame(main_container, bg="#ffffff", relief="flat", width=400)
            right_panel.pack(side="right", fill="y", padx=(10, 0))
            right_panel.pack_propagate(False)
            
            # Título del panel de análisis
            analysis_title = tk.Label(
                right_panel,
                text="📊 Análisis en Tiempo Real",
                font=("Arial", 16, "bold"),
                bg="#ffffff",
                fg="#1e293b"
            )
            analysis_title.pack(pady=20)
            
            # Estado actual
            self.status_frame = tk.Frame(right_panel, bg="#ffffff")
            self.status_frame.pack(fill="x", padx=20, pady=10)
            
            self.current_emotion_var = tk.StringVar(value="---")
            self.current_emotion_label = tk.Label(
                self.status_frame,
                text="Emoción actual:",
                font=("Arial", 12, "bold"),
                bg="#ffffff",
                fg="#374151"
            )
            self.current_emotion_label.pack(anchor="w")
            
            self.emotion_display = tk.Label(
                self.status_frame,
                textvariable=self.current_emotion_var,
                font=("Arial", 18, "bold"),
                bg="#ffffff",
                fg="#10b981"
            )
            self.emotion_display.pack(anchor="w", pady=(5, 0))
            
            # Frame para recomendaciones
            self.recommendation_frame = tk.Frame(right_panel, bg="#f8fafc", relief="flat")
            self.recommendation_frame.pack(fill="x", padx=20, pady=20)
            
            rec_title = tk.Label(
                self.recommendation_frame,
                text="💡 Recomendación",
                font=("Arial", 12, "bold"),
                bg="#f8fafc",
                fg="#374151"
            )
            rec_title.pack(pady=(15, 5))
            
            self.recommendation_var = tk.StringVar(value="Inicia la sesión para obtener recomendaciones")
            self.recommendation_label = tk.Label(
                self.recommendation_frame,
                textvariable=self.recommendation_var,
                font=("Arial", 10),
                bg="#f8fafc",
                fg="#6b7280",
                wraplength=350,
                justify="center"
            )
            self.recommendation_label.pack(pady=(0, 15))
            
            # Frame para gráfico
            self.chart_frame = tk.Frame(right_panel, bg="#ffffff")
            self.chart_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            chart_title = tk.Label(
                self.chart_frame,
                text="📈 Distribución de Emociones",
                font=("Arial", 12, "bold"),
                bg="#ffffff",
                fg="#374151"
            )
            chart_title.pack(pady=(0, 10))
            
            # Configurar matplotlib
            self.fig, self.ax = plt.subplots(figsize=(5, 3), facecolor='white')
            self.canvas = FigureCanvasTkAgg(self.fig, self.chart_frame)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            
            # Estadísticas de sesión
            self.stats_frame = tk.Frame(right_panel, bg="#ffffff")
            self.stats_frame.pack(fill="x", padx=20, pady=20)
            
            stats_title = tk.Label(
                self.stats_frame,
                text="📋 Estadísticas de Sesión",
                font=("Arial", 12, "bold"),
                bg="#ffffff",
                fg="#374151"
            )
            stats_title.pack(anchor="w")
            
            self.stats_text = tk.StringVar(value="Duración: 00:00\nDetecciones: 0")
            stats_display = tk.Label(
                self.stats_frame,
                textvariable=self.stats_text,
                font=("Arial", 10),
                bg="#ffffff",
                fg="#6b7280",
                justify="left"
            )
            stats_display.pack(anchor="w", pady=(5, 0))
            
            # Footer
            footer_frame = tk.Frame(self.ventana, bg="#f8fafc", height=50)
            footer_frame.pack(fill="x")
            footer_frame.pack_propagate(False)
            
            exit_btn = tk.Button(
                footer_frame,
                text="❌ Cerrar Aplicación",
                font=("Arial", 10),
                bg="#ef4444",
                fg="white",
                relief="flat",
                cursor="hand2",
                command=self.salir,
                width=20,
                pady=8
            )
            exit_btn.pack(pady=10)
            
        except Exception as e:
            print(f"Error al crear widgets: {e}")
            messagebox.showerror("Error", f"Error al crear la interfaz: {e}")
        
    def create_control_button(self, parent, text, color_normal, color_hover, command):
        """Crear botones de control con efectos hover"""
        try:
            btn = tk.Button(
                parent,
                text=text,
                font=("Arial", 10, "bold"),
                bg=color_normal,
                fg="white",
                relief="flat",
                cursor="hand2",
                command=command,
                width=12,
                pady=10,
                activebackground=color_hover,
                activeforeground="white",
                bd=0
            )
            
            # Efectos hover
            def on_enter(e):
                btn.configure(bg=color_hover)
                
            def on_leave(e):
                btn.configure(bg=color_normal)
                
            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)
            
            return btn
        except Exception as e:
            print(f"Error creando botón: {e}")
            return None
        
    def predecir_emocion(self, rostro):
        """Predecir emoción de un rostro"""
        try:
            rostro = cv2.resize(rostro, (48, 48))
            rostro = rostro.astype('float32') / 255.0
            rostro = np.expand_dims(rostro, axis=0)
            rostro = np.expand_dims(rostro, axis=-1)
            pred = modelo.predict(rostro, verbose=0)
            emocion = emociones[np.argmax(pred)]
            return emocion
        except Exception as e:
            print(f"Error en predicción: {e}")
            return "neutral"
        
    def update_chart(self):
        """Actualizar gráfico de distribución de emociones"""
        try:
            self.ax.clear()
            
            if not self.emociones_detectadas:
                self.ax.text(0.5, 0.5, 'Esperando datos...', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=self.ax.transAxes, fontsize=10, color='#6b7280')
            else:
                conteo = Counter(self.emociones_detectadas)
                emotions = list(conteo.keys())
                counts = list(conteo.values())
                colors = [emotion_colors.get(emo, '#6b7280') for emo in emotions]
                
                bars = self.ax.bar(emotions, counts, color=colors, alpha=0.8)
                
                # Estilizar el gráfico
                self.ax.set_xlabel('Emociones', fontsize=8, color='#374151')
                self.ax.set_ylabel('Frecuencia', fontsize=8, color='#374151')
                self.ax.tick_params(colors='#6b7280', labelsize=7)
                self.ax.spines['top'].set_visible(False)
                self.ax.spines['right'].set_visible(False)
                self.ax.spines['bottom'].set_color('#e2e8f0')
                self.ax.spines['left'].set_color('#e2e8f0')
                self.ax.grid(True, alpha=0.3)
                
                # Agregar valores en las barras
                for bar in bars:
                    height = bar.get_height()
                    self.ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                               f'{int(height)}', ha='center', va='bottom', fontsize=7)
                    
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            print(f"Error actualizando gráfico: {e}")
        
    def actualizar_camara(self):
        """Función principal para capturar y procesar video"""
        try:
            detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.start_time = datetime.now()
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                self.frame_count += 1
                
                # Redimensionar frame para mejor rendimiento
                frame = cv2.resize(frame, (800, 600))
                gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rostros = detector.detectMultiScale(gris, 1.3, 5)
                
                current_emotions = []
                for (x, y, w, h) in rostros:
                    rostro = gris[y:y+h, x:x+w]
                    emocion = self.predecir_emocion(rostro)
                    current_emotions.append(emocion)
                    
                    # Convertir color hex a BGR para OpenCV
                    color_hex = emotion_colors.get(emocion, '#6b7280')
                    color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
                    color_bgr = color_rgb[::-1]  # BGR para OpenCV
                    
                    # Dibujar rectángulo con estilo moderno
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color_bgr, 3)
                    
                    # Texto con fondo de color
                    cv2.rectangle(frame, (x, y-40), (x+w, y), color_bgr, -1)
                    cv2.putText(frame, emocion.upper(), (x+5, y-15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                # Actualizar datos si se detectaron emociones
                if current_emotions:
                    main_emotion = max(set(current_emotions), key=current_emotions.count)
                    self.emociones_detectadas.extend(current_emotions)
                    self.current_emotion_var.set(main_emotion.upper())
                    self.emotion_display.configure(fg=emotion_colors.get(main_emotion, '#6b7280'))
                    self.recommendation_var.set(recomendaciones.get(main_emotion, "---"))
                    
                # Actualizar estadísticas de tiempo
                if self.start_time:
                    elapsed_time = datetime.now() - self.start_time
                    minutes, seconds = divmod(elapsed_time.seconds, 60)
                    self.stats_text.set(f"Duración: {minutes:02d}:{seconds:02d}\nDetecciones: {len(self.emociones_detectadas)}")
                
                # Mostrar video en la interfaz
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.configure(image=imgtk, text="")
                self.video_label.image = imgtk
                
                # Actualizar gráfico cada 30 frames para mejor rendimiento
                if self.frame_count % 30 == 0:
                    self.update_chart()
                    
                # Actualizar interfaz
                try:
                    self.ventana.update()
                except tk.TclError:
                    # Ventana cerrada, salir del bucle
                    break
                    
        except Exception as e:
            print(f"Error en cámara: {e}")
        finally:
            # Limpiar cuando se detiene
            if self.cap:
                self.cap.release()
            try:
                self.video_label.configure(image="", text="📹 Video detenido")
            except:
                pass
        
    def iniciar(self):
        """Iniciar captura de video y análisis"""
        self.emociones_detectadas.clear()
        self.frame_count = 0
        
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "No se pudo acceder a la cámara")
                return
                
            self.running = True
            self.current_emotion_var.set("Iniciando...")
            threading.Thread(target=self.actualizar_camara, daemon=True).start()
            
            # Actualizar estado de botones
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al iniciar: {str(e)}")
            
    def finalizar(self):
        """Detener análisis y mostrar resumen"""
        self.running = False
        if self.cap:
            self.cap.release()
            
        # Restaurar estado de botones
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        
        # Actualizar interfaz
        self.current_emotion_var.set("Detenido")
        self.emotion_display.configure(fg="#6b7280")
        self.recommendation_var.set("Sesión finalizada")
        
        # Mostrar resumen si hay datos
        if self.emociones_detectadas:
            self.mostrar_resumen()
            
    def guardar_sesion(self):
        """Guardar reporte detallado y gráficos de la sesión"""
        if not self.emociones_detectadas:
            messagebox.showerror("Error", "No hay datos de sesión para guardar")
            return
            
        try:
            conteo = Counter(self.emociones_detectadas)
            emocion_principal = conteo.most_common(1)[0][0]
            
            # Crear directorio results si no existe
            os.makedirs("results", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Guardar reporte detallado
            ruta_reporte = f"results/sesion_detallada_{timestamp}.txt"
            with open(ruta_reporte, "w", encoding="utf-8") as f:
                f.write("🧠 REPORTE DE SESIÓN - ANÁLISIS EMOCIONAL\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total de detecciones: {len(self.emociones_detectadas)}\n\n")
                
                f.write("DISTRIBUCIÓN DE EMOCIONES:\n")
                f.write("-" * 25 + "\n")
                for emo, count in conteo.most_common():
                    porcentaje = (count / len(self.emociones_detectadas)) * 100
                    f.write(f"{emo.capitalize():<12}: {count:>3} ({porcentaje:>5.1f}%)\n")
                
                f.write(f"\nEMOCIÓN PREDOMINANTE: {emocion_principal.upper()}\n")
                f.write(f"RECOMENDACIÓN: {recomendaciones.get(emocion_principal, 'N/A')}\n\n")
                
                # Análisis de sentiment
                emociones_positivas = ['happy', 'surprise']
                emociones_negativas = ['angry', 'sad', 'fear', 'disgust']
                
                positivas = sum(conteo.get(emo, 0) for emo in emociones_positivas)
                negativas = sum(conteo.get(emo, 0) for emo in emociones_negativas)
                neutrales = conteo.get('neutral', 0)
                
                total = positivas + negativas + neutrales
                if total > 0:
                    f.write("ANÁLISIS GENERAL:\n")
                    f.write("-" * 16 + "\n")
                    f.write(f"Emociones positivas: {(positivas/total)*100:.1f}%\n")
                    f.write(f"Emociones negativas: {(negativas/total)*100:.1f}%\n")
                    f.write(f"Emociones neutrales: {(neutrales/total)*100:.1f}%\n")
            
            # Guardar gráfico
            ruta_grafico = f"results/grafico_sesion_{timestamp}.png"
            self.guardar_grafico(ruta_grafico, conteo)
            
            messagebox.showinfo("Éxito", f"Sesión guardada exitosamente:\n• Reporte: {ruta_reporte}\n• Gráfico: {ruta_grafico}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar sesión: {str(e)}")
            
    def guardar_grafico(self, ruta, conteo):
        """Generar y guardar gráficos detallados"""
        try:
            # Crear figura con dos subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
            
            # Preparar datos
            emotions = list(conteo.keys())
            counts = list(conteo.values())
            colors = [emotion_colors.get(emo, '#6b7280') for emo in emotions]
            
            # Gráfico de barras
            bars = ax1.bar(emotions, counts, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
            ax1.set_title('Distribución de Emociones', fontsize=14, fontweight='bold', color='#1e293b')
            ax1.set_xlabel('Emociones', fontsize=10, color='#374151')
            ax1.set_ylabel('Frecuencia', fontsize=10, color='#374151')
            
            # Agregar valores encima de las barras
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Gráfico circular
            ax2.pie(counts, labels=emotions, colors=colors, autopct='%1.1f%%', 
                   startangle=90, textprops={'fontsize': 9})
            ax2.set_title('Proporción de Emociones', fontsize=14, fontweight='bold', color='#1e293b')
            
            # Mejorar estilo
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['bottom'].set_color('#e2e8f0')
            ax1.spines['left'].set_color('#e2e8f0')
            ax1.tick_params(colors='#6b7280')
            ax1.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(ruta, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
        except Exception as e:
            print(f"Error guardando gráfico: {e}")
        
    def mostrar_resumen(self):
        """Mostrar ventana de resumen de sesión"""
        try:
            conteo = Counter(self.emociones_detectadas)
            emocion_principal = conteo.most_common(1)[0][0]
            
            # Crear mensaje de resumen
            resumen_texto = f"Sesión completada exitosamente!\n\n"
            resumen_texto += f"Emoción predominante: {emocion_principal.upper()}\n"
            resumen_texto += f"Recomendación: {recomendaciones.get(emocion_principal, '')}\n\n"
            resumen_texto += "Top 3 emociones detectadas:\n"
            
            for i, (emo, count) in enumerate(conteo.most_common(3)):
                porcentaje = (count / len(self.emociones_detectadas)) * 100
                resumen_texto += f"{i+1}. {emo.capitalize()}: {count} ({porcentaje:.1f}%)\n"
              #enviar correo de sesion
            enviar_recomendacion_por_correo(emocion_principal, recomendaciones.get(emocion_principal, "Sin recomendación."))
            messagebox.showinfo("Resumen de Sesión", resumen_texto)
            
        except Exception as e:
            print(f"Error mostrando resumen: {e}")
            messagebox.showinfo("Resumen", "Sesión completada.")
        
    def salir(self):
        """Cerrar aplicación"""
        if self.running:
            self.finalizar()
        self.ventana.destroy()

# Función principal
def main():
    try:
        ventana = tk.Tk()
        app = ModernEmotionApp(ventana)
        ventana.mainloop()
    except Exception as e:
        print(f"Error iniciando aplicación: {e}")
        messagebox.showerror("Error", f"Error iniciando aplicación: {e}")

if __name__ == "__main__":
    main()
