import cv2
import os
import numpy as np
from PIL import Image
import pickle  # Для словаря

path = os.path.dirname(os.path.abspath(__file__))
recognizer = cv2.face.LBPHFaceRecognizer_create()
faceCascade = cv2.CascadeClassifier(path + r'/trainer/haarcascade_frontalface_default.xml')
dataPath = path + r'/Faces' # путь к датасету

# Получить изображения и метки (числовые)
def get_images_and_labels(datapath):
    image_paths = [os.path.join(datapath, f) for f in os.listdir(datapath)]
    images = []
    labels = []
    labels_to_names = {}  # Словарь id -> имя
    next_id = 0  # Уникальный ID для каждого имени

    for image_path in image_paths:
        # Чтение изображения и преобразование в оттенки серого
        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, 'uint8')

        # Извлечение имени из названия файла
        name = os.path.split(image_path)[1].split(".")[0].replace("face-", "")
        
        # Если имя новое, добавляем его в словарь
        if name not in labels_to_names.values():
            labels_to_names[next_id] = name
            label = next_id
            next_id += 1
        else:
            label = list(labels_to_names.keys())[list(labels_to_names.values()).index(name)]

        # Обнаружение лица
        faces = faceCascade.detectMultiScale(image)
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(label)
            cv2.imshow("Adding faces to training set...", image[y: y + h, x: x + w])
            cv2.waitKey(10)

    # Сохранение словаря id -> имя
    with open(os.path.join(path, 'trainer/labels_to_names.pkl'), 'wb') as f:
        pickle.dump(labels_to_names, f)

    return images, labels

# Получение изображений и меток
images, labels = get_images_and_labels(dataPath)
labels = np.array(labels, dtype='int')

# Обучение модели
recognizer.train(images, labels)
if not os.path.exists(os.path.join(path, 'trainer')):
    os.makedirs(os.path.join(path, 'trainer'))
recognizer.save(os.path.join(path, 'trainer/trainer.yml'))

cv2.destroyAllWindows()
