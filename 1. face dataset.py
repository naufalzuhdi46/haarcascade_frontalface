import cv2
import os

# Membuat direktori 'dataset' jika belum ada
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Membuka webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set lebar video
cam.set(4, 480)  # set tinggi video

# Menggunakan haarcascade untuk deteksi wajah
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Meminta user ID (string)
face_id = input('\n enter user id (string) and press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look at the camera and wait ...")
count = 0

# Mulai menangkap gambar
while True:
    ret, img = cam.read()  # Ambil gambar dari webcam
    faces = face_detector.detectMultiScale(img, 1.3, 5)  # Deteksi wajah dari gambar berwarna

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Gambar kotak sekitar wajah
        count += 1

        # Simpan gambar yang ditangkap ke folder 'dataset' dalam format berwarna
        cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", img[y:y + h, x:x + w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff  # Tekan 'ESC' untuk keluar dari video
    if k == 27:  # Tombol ESC memiliki kode ASCII 27
        break
    elif count >= 100:  # Mengambil 20 gambar wajah
        break

# Membersihkan
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
