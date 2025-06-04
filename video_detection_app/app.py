import os
import time
import cv2
import smtplib
import numpy as np
import threading
import queue
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email import encoders
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO
from ultralytics import YOLO
import threading

# ---------- Configuração SMTP ----------
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "@gmail.com"
SMTP_PASSWORD = os.getenv("SMTP_PASS", "pass")
FROM_EMAIL = SMTP_USER
TO_EMAIL = "@gmail.com"

recorder = None

# ---------- Função responsável por enviar os alertas e video por e-mail ----------
def send_email(subject: str, body: str, attachments: list[str] = None):
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = FROM_EMAIL
    msg["To"] = TO_EMAIL
    msg.attach(MIMEText(body, "plain"))
    if attachments:
        for filepath in attachments:
            if not os.path.isfile(filepath):
                continue
            part = MIMEBase("application", "octet-stream")
            with open(filepath, "rb") as f:
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f'attachment; filename="{os.path.basename(filepath)}"'
            )
            msg.attach(part)
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)

# ---------- Aplicação Flask + SocketIO ----------
app = Flask(__name__)
socketio = SocketIO(app)

# ---------- Rota principal da aplicação Flask que carrega a página inicial ----------
@app.route('/')
def index():
    return render_template('index.html')

# ---------- Rota que lista os vídeos gravados na pasta de capturas ----------
@app.route('/capturas')
def capturas():
    pasta = os.path.join("static", "capturas")
    os.makedirs(pasta, exist_ok=True)
    videos = [f for f in os.listdir(pasta) if f.endswith(".webm")]
    videos.sort(reverse=True)
    return render_template('capturas.html', imagens=videos)

# ---------- Rota para apagar um vídeo gravado ----------
@app.route('/delete/<filename>', methods=['DELETE'])
def delete_video(filename):
    caminho = os.path.join("static", "capturas", filename)
    try:
        if os.path.exists(caminho):
            os.remove(caminho)
            return jsonify({'success': True}), 200
        else:
            return jsonify({'error': 'Ficheiro não encontrado'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# ---------- Função auxiliar para verificar se a caixa delimitadora está dentro de uma zona específica ----------
def verificar_zona(x1, y1, x2, y2, zona):
    zx1, zy1, zx2, zy2 = zona
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2

# ---------- Classe responsável pela gravação do vídeo em segundo plano ----------
class VideoRecorder(threading.Thread):
    def __init__(self, video_path, fps, size):
        super().__init__()
        self.video_path = video_path
        self.fps = fps
        self.original_size = size
        self.reduced_size = (size[0] // 2, size[1] // 2)
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'VP80'), fps, self.reduced_size)

    def run(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
                small_frame = cv2.resize(frame, self.reduced_size)
                self.writer.write(small_frame)
            except queue.Empty:
                continue
        self.writer.release()
        send_email(subject="Safecam: Vídeo de Alerta",
                   body="Segue em anexo o vídeo de alerta gravado.",
                   attachments=[self.video_path])
        print("[INFO] Gravação terminada.")
        global recorder
        recorder = None

    def add_frame(self, frame):
        self.queue.put(frame)

    def stop(self):
        self.stop_event.set()

# ---------- Função principal de captura e processamento de vídeo (webcam ou ficheiro) ----------
def gen_frames(source="pessoaEcarro.mp4"):
    model = YOLO("yolo11n.pt")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    is_webcam = isinstance(source, int) or (isinstance(source, str) and source.isdigit())
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    TARGET_FPS = 5 if is_webcam else (original_fps if original_fps > 0 else 30)
    FRAME_INTERVAL = 1.0 / TARGET_FPS
    last_frame_time = time.time()

    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    zona1 = (0, 0, largura // 3, altura)
    zona2 = (largura // 3, 0, 2 * largura // 3, altura)
    zona3 = (2 * largura // 3, 0, largura, altura)
    tempo_limite = 2
    alerta_pessoa = False
    alerta_carro = False
    tempo_pessoa = 0
    tempo_carro = 0

    global recorder
    video_path = None

    pending_alerts = []  
    pending_ends = []   

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            if is_webcam:
                now = time.time()
                if now - last_frame_time < FRAME_INTERVAL:
                    continue
                last_frame_time = now

            results = model(frame, verbose=False)[0]

            num_pessoas = 0
            num_carros = 0

            for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                confidence = float(conf) * 100
                if confidence < 65:
                    continue  

                cls = int(cls_id)
                x1, y1, x2, y2 = map(int, box)

                if cls == 0:
                    num_pessoas += 1
                    tipo, cor = "Pessoa", (0, 255, 0)
                elif cls == 2:
                    num_carros += 1
                    tipo, cor = "Carro", (255, 0, 0)
                else:
                    continue

                label = f"{tipo} {confidence:.0f}%"
                cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)


            zonas = [zona1, zona2, zona3]
            zona_alerta = [False, False, False]
            for box, cls_id in zip(results.boxes.xyxy, results.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                for i, zona in enumerate(zonas):
                    if verificar_zona(x1, y1, x2, y2, zona):
                        zona_alerta[i] = True

            x1_div = largura // 3
            x2_div = 2 * largura // 3
            cor_linha = (200, 200, 200)
            cor1 = (0, 0, 255) if zona_alerta[0] else cor_linha
            cor2 = (0, 0, 255) if zona_alerta[1] else cor_linha
            cor3 = (0, 0, 255) if zona_alerta[2] else cor_linha

            cv2.line(frame, (x1_div, 0), (x1_div, altura), cor_linha, 2)
            cv2.line(frame, (x2_div, 0), (x2_div, altura), cor_linha, 2)

            cv2.putText(frame, "Zona 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor1, 2)
            cv2.putText(frame, "Zona 2", (x1_div + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor2, 2)
            cv2.putText(frame, "Zona 3", (x2_div + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor3, 2)

            contador_texto = f"Pessoas: {num_pessoas}  |  Carros: {num_carros}"
            (text_width, _), _ = cv2.getTextSize(contador_texto, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            x = frame.shape[1] - text_width - 10
            y = frame.shape[0] - 10
            cv2.putText(frame, contador_texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            if num_pessoas > 0:
                if not alerta_pessoa:
                    pending_alerts.append({'tipo': 'pessoa', 'message': f"{num_pessoas} pessoa(s) detetadas!"})
                    send_email(subject="Safecam: Pessoa(s) Detetada(s)",
                               body=f"Detetámos {num_pessoas} pessoa(s) em {time.strftime('%Y-%m-%d %H:%M:%S')}.")
                    alerta_pessoa = True
                tempo_pessoa = 0
            else:
                tempo_pessoa += 1
                if alerta_pessoa and tempo_pessoa >= TARGET_FPS * tempo_limite:
                    pending_ends.append({'tipo': 'pessoa'})
                    alerta_pessoa = False

            if num_carros > 0:
                if not alerta_carro:
                    pending_alerts.append({'tipo': 'carro', 'message': f"{num_carros} carro(s) detetados!"})
                    send_email(subject="Safecam: Carro(s) Detetado(s)",
                               body=f"Detetámos {num_carros} carro(s) em {time.strftime('%Y-%m-%d %H:%M:%S')}.")
                    alerta_carro = True
                tempo_carro = 0
            else:
                tempo_carro += 1
                if alerta_carro and tempo_carro >= TARGET_FPS * tempo_limite:
                    pending_ends.append({'tipo': 'carro'})
                    alerta_carro = False

            if (num_pessoas > 0 or num_carros > 0) and recorder is None:
                nome = f"captura_{int(time.time())}.webm"
                video_path = os.path.join("static", "capturas", nome)
                os.makedirs(os.path.dirname(video_path), exist_ok=True)
                recorder = VideoRecorder(video_path, TARGET_FPS, (frame.shape[1], frame.shape[0]))
                recorder.start()
                print(f"[INFO] Iniciada gravação: {nome}")
            elif (num_pessoas == 0 and num_carros == 0) and recorder:
                if tempo_pessoa >= TARGET_FPS * tempo_limite and tempo_carro >= TARGET_FPS * tempo_limite:
                    def delayed_stop(local_recorder):
                        time.sleep(1)
                        local_recorder.stop()
                        local_recorder.join()
                    threading.Thread(target=delayed_stop, args=(recorder,)).start()

            if recorder:
                recorder.add_frame(frame)

            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            for alert in pending_alerts:
                socketio.emit('alert', alert)
            pending_alerts.clear()

            for end in pending_ends:
                socketio.emit('alert_end', end)
            pending_ends.clear()


    except GeneratorExit:
        print("[INFO] Cliente desconectou. Forçando finalização...")
        if recorder:
            recorder.stop()
            recorder.join()
            recorder = None
        pass  

    finally:
        cap.release()
        if recorder:
            print("[INFO] Aguardando finalização da gravação...")
            recorder.stop()
            recorder.join() 
            recorder = None

        if alerta_pessoa:
            socketio.emit('alert_end', {'tipo': 'pessoa'})
        if alerta_carro:
            socketio.emit('alert_end', {'tipo': 'carro'})

# ---------- Rota para o feed de vídeo em tempo real ----------          
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------- Início da aplicação com Flask e SocketIO ----------
if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0')
