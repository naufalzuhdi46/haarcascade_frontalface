import cv2
import numpy as np
import os
import gradio as gr

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')  # load trained model
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# Load name to ID mapping
name_to_id = {}
id_to_name = {}
with open('trainer/name_to_id.txt', 'r') as f:
    for line in f:
        name, id = line.strip().split(',')
        name_to_id[name] = int(id)
        id_to_name[int(id)] = name

# Define min window size to be recognized as a face
minW = 0.1 * 640  # set video width
minH = 0.1 * 480  # set video height

def detect_faces(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # Print the id and confidence to debug
            print(f"Detected ID: {id}, Confidence: {confidence}")

            # Check if confidence is less than 100 ==> "0" is perfect match
            if confidence < 100:
                name = id_to_name.get(id, "tidakada")
                confidence_text = "  {0}%".format(round(140 - confidence))
                if name == "tidakada":
                    color = (0, 0, 255)  # Red for unknown
                else:
                    color = (0, 255, 0)  # Green for known
            else:
                name = "tidakada"
                confidence_text = "  {0}%".format(round(100 - confidence))
                color = (0, 0, 255)  # Red for unknown

            # Draw rectangle around the face
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            # Display the name and confidence
            cv2.putText(image, name, (x + 5, y - 5), font, 1, color, 2)
            cv2.putText(image, confidence_text, (x + 5, y + h - 5), font, 1, color, 1)

        return image
    except Exception as e:
        print(f"Error in detect_faces: {e}")
        return image

# Create Gradio interface
iface = gr.Interface(fn=detect_faces, inputs="image", outputs="image")
iface.launch()