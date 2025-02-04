import cv2
import os, qr_gen

path = os.path.dirname(os.path.abspath(__file__))
detector = cv2.CascadeClassifier(path + r'/trainer/haarcascade_frontalface_default.xml')

i=0 # счётчик изображений
offset=50 # расстояния от распознанного лица до рамки
name=input('Введите имя пользователя: ')
video=cv2.VideoCapture(0)

while True:
    ret, im =video.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
    # Обработка лиц
    for(x,y,w,h) in faces:
        # Защита от выхода за границы изображения
        x_start = max(0, x - offset)
        y_start = max(0, y - offset)
        x_end = min(im.shape[1], x + w + offset)
        y_end = min(im.shape[0], y + h + offset)

        face_region = gray[y_start:y_end, x_start:x_end]

        # Сохранение лица
        if face_region.size > 0:
            i=i+1
            cv2.imwrite(path + r'/Faces/face-' + name + '.' + str(i) + ".jpg", face_region)
            cv2.rectangle(im, (x_start, y_start), (x_end, y_end), (225, 0, 0), 2)
            cv2.imshow('im', face_region)
            cv2.waitKey(50)  # пауза перед новым кадром
    # если у нас хватает кадров
    if i>100:
        qr_gen.generate_qr(name, name)
        video.release()
        cv2.destroyAllWindows()
        break