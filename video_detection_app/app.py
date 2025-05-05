import os
import time
import cv2
import numpy as np
import threading
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
from ultralytics import YOLO

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capturas')
def capturas():
    pasta = os.path.join("static", "capturas")
    videos = [f for f in os.listdir(pasta) if f.endswith(".webm")]
    videos.sort(reverse=True)
    return render_template("capturas.html", imagens=videos)

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

def gen_frames():
    model = YOLO("yolo11n.pt")
    cap = cv2.VideoCapture("video3.mp4")

    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = None
    gravando = False
    tempo_sem_objetos = 0
    tempo_maximo_sem_obj = 2  # segundos
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    previous_boxes_by_type = {}

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, verbose=False)[0]
        current_time = time.time()
        detected = False
        detected_types = set()

        for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            cls = int(cls_id)
            confidence = float(conf) * 100
            x1, y1, x2, y2 = map(int, box)
            nova_caixa = np.array([x1, y1, x2, y2])

            tipo, color = None, (0, 255, 0)
            if cls == 0:
                tipo = "Pessoa"; color = (0, 255, 0)
            elif cls == 2:
                tipo = "Carro"; color = (255, 0, 0)

            if tipo:
                label = f"{tipo} {confidence:.0f}%"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                caixas_anteriores = previous_boxes_by_type.get(tipo, [])
                novo_objeto = all(np.linalg.norm(nova_caixa - np.array(c)) > 60 for c in caixas_anteriores)

                if novo_objeto:
                    detected_types.add(tipo)
                    previous_boxes_by_type.setdefault(tipo, []).append([x1, y1, x2, y2])

                detected = True

        if detected:
            if not gravando:
                nome_video = f"captura_{int(current_time)}.webm"
                caminho_video = os.path.join("static", "capturas", nome_video)
                os.makedirs("static/capturas", exist_ok=True)
                out = cv2.VideoWriter(caminho_video, fourcc, fps, (frame_largura, frame_altura))
                print(f"[INFO] Iniciada gravação: {nome_video}")
                for tipo in detected_types:
                    socketio.emit('alert', {'message': f"{tipo} detetado!"})
                gravando = True
            tempo_sem_objetos = 0
        else:
            if gravando:
                tempo_sem_objetos += 1
                if tempo_sem_objetos >= fps * tempo_maximo_sem_obj:
                    out.release()
                    print("[INFO] Gravação terminada.")
                    gravando = False
                    tempo_sem_objetos = 0

        if gravando and out:
            out.write(frame)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0')
