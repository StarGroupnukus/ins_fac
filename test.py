from datetime import datetime
import cv2
import os
from ultralytics import YOLO
import time
from funcs import is_nose_inside
import torch
from insightface.app import FaceAnalysis

class VideoProcessor:
    def __init__(self, camera_url, group, app):
        self.camera_url = camera_url
        self.users = []
        self.app = app
        self.group = f'group-{group}_ID'
        self.ids_dict = {}

    def process_video(self):
        cap = cv2.VideoCapture(self.camera_url)
        if not cap.isOpened():
            print("Не удалось подключиться к камере.")
            return

        os.makedirs('screenshots', exist_ok=True)
        number = 0

        try:
            while True:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    print("Не удалось получить кадр.")
                    break

                frame = cv2.resize(frame, (1280, 720))

                faces = self.app.get(frame)

                anotation_frame = self.draw_rectangle(frame, faces)

                #cv2.imshow(self.group, anotation_frame)
                number += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                print(f"Frame Processing Time: {(time.time() - start_time) * 1000} ms")
        finally:
            cap.release()

    def draw_rectangle(self, frame, faces):
        for face in faces:
            if is_nose_inside(face) and face['det_score'] > 0.80:
                # Рисуем прямоугольник вокруг лица
                x1, y1, x2, y2 = [int(value) for value in face['bbox']]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Отрисовываем ключевые точки
                for point in face['kps']:
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)  # Красный цвет для ключевых точек

                # Отображаем det_score
                det_score = face['det_score']
                score_text = f"Score: {det_score:.2f}"
                cv2.putText(frame, score_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return frame


if __name__ == "__main__":
    USERNAME = 'admin'
    PASSWORD = 'Babur2001'
    camera_url = f'rtsp://{USERNAME}:{PASSWORD}@192.168.0.119:554/Streaming/Channels/101'
    group = 1
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], allowed_modules=['detection'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    processor = VideoProcessor(camera_url=camera_url, group=group, app=app)
    processor.process_video()
