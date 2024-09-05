import cv2
import numpy as np
from PIL import Image
import os

# Path untuk menyimpan database wajah
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Fungsi untuk menormalisasi gambar
def normalize_images(images):
    norm_images = []
    for img in images:
        norm_img = cv2.equalizeHist(img)
        norm_images.append(norm_img)
    return norm_images

# Fungsi untuk mendapatkan gambar dan data label
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    name_to_id = {}
    current_id = 0

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # convert to grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        name = os.path.split(imagePath)[-1].split(".")[1]
        if name not in name_to_id:
            name_to_id[name] = current_id
            current_id += 1
        id = name_to_id[name]
        
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    faceSamples = normalize_images(faceSamples)  # Normalize images
    return faceSamples, ids, name_to_id

print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids, name_to_id = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Menyimpan model ke trainer/trainer.yml
recognizer.write('trainer/trainer.yml')

# Menyimpan mapping name_to_id
if not os.path.exists('trainer'):
    os.makedirs('trainer')

with open('trainer/name_to_id.txt', 'w') as f:
    for name, id in name_to_id.items():
        f.write(f"{name},{id}\n")

# Menampilkan jumlah wajah yang dilatih dan mengakhiri program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
