import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def enviar_recomendacion_por_correo(emocion, recomendacion):
    # Configuración
    remitente = "luissebastiansanchez15@gmail.com"
    app_password = "svxg vysp kufk qthg" 
    receptor = "suissebas12@gmail.com"
    asunto = "Recomendación emocional en clase"
    
    # Cuerpo del mensaje
    mensaje = f"""
    Estimado catedrático,

    Se ha detectado que la emoción predominante en clase es: {emocion.upper()}.

    Recomendación sugerida:
    → {recomendacion}

    Atentamente,
    Sistema P2IA-FACE2FEEDBACK
    """

    # Preparar el mensaje MIME
    msg = MIMEMultipart()
    msg['From'] = remitente
    msg['To'] = receptor
    msg['Subject'] = asunto
    msg.attach(MIMEText(mensaje, 'plain'))

    try:
        servidor = smtplib.SMTP('smtp.gmail.com', 587)
        servidor.starttls()
        servidor.login(remitente, app_password)
        servidor.send_message(msg)
        servidor.quit()
        print("✅ Correo enviado exitosamente.")
    except Exception as e:
        print("❌ Error al enviar correo:", str(e))
