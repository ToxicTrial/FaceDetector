import cv2
import os
import pickle  # Для загрузки словаря 
from pyzbar.pyzbar import decode  # Для декодирования QR-кодов
import numpy as np
from PIL import Image, ImageDraw, ImageFont # Для отображения кириллицы

path = os.path.dirname(os.path.abspath(__file__))
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(path + r'/trainer/trainer.yml') # модель распознавания
faceCascade = cv2.CascadeClassifier(path + r'/trainer/haarcascade_frontalface_default.xml')

# загружаем словарь ID -> имена
with open(path + '/trainer/labels_to_names.pkl', 'rb') as f:
    labels_to_names = pickle.load(f)

cam = cv2.VideoCapture(0) # доступ к камере
font = cv2.FONT_HERSHEY_SIMPLEX # шрифт для вывода подписей

threshold = 70

qr_user_data = None
face_user_name = None

# функция для добавления текста с кириллицей
def put_text_with_pillow(img, text, position, font_size=30, color=(0, 255, 0)):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)  # Путь к шрифту
    except IOError:
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img

while True:
    ret, im = cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

    qr_user_data = None
    face_user_name = None

    for (x, y, w, h) in faces:
        nbr_predicted, coord = recognizer.predict(gray[y:y+h, x:x+w])
        print(coord)
        if coord > threshold:
            name = "Unknown"
        else:
            name = labels_to_names.get(nbr_predicted, "Unknown")
            face_user_name = name
        cv2.rectangle(im, (x-50, y-50), (x+w+50, y+h+50), (225, 0, 0), 2) # прямоугольник вокруг лица
        im = put_text_with_pillow(im, name, (x, y + h + 30), font_size=30, color=(0, 255, 0))
    
    qr_codes = decode(im)  # Декодируем QR-коды
    for qr_code in qr_codes:
        qr_data = qr_code.data.decode('utf-8')
        # прямоугольник вокруг QR-кода
        rect_points = qr_code.polygon
        if len(rect_points) == 4:
            pts = rect_points
        else:
            pts = qr_code.rect
            pts = [pts]
        pts = cv2.convexHull(np.array(pts, dtype=np.int32))
        cv2.polylines(im, [pts], True, (255, 0, 255), 3)
        im = put_text_with_pillow(im, qr_data, (qr_code.rect[0], qr_code.rect[1] - 20), font_size=20, color=(0, 255, 0))
        qr_user_data = qr_data  

    if qr_user_data and face_user_name:
        if qr_user_data == face_user_name:
            im = put_text_with_pillow(im, "Доступ разрешён", (50, 50), font_size=16, color=(0, 255, 0))
        else:
            im = put_text_with_pillow(im, "Доступ запрещён", (50, 50), font_size=16, color=(255, 0, 0))
    
    cv2.imshow('Face recognition', im)
    cv2.imshow('Gray', gray)

    if cv2.waitKey(10) & 0xFF == ord('q'):  # 'q', чтобы выйти
        break

cam.release()
cv2.destroyAllWindows()
