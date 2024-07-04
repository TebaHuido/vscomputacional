import cv2
import face_recognition
import numpy as np
import pickle

# Cargar el modelo SVM
model_path = './generated-embeddings/classifier.pkl'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)
print("Modelo SVM cargado.")

# Definir la ruta de los datos y las personas
persons = ['Copito', 'Nacho', 'Seba H', 'Nico V', 'Roby']
print('Personas:', persons)

# Inicializar la cámara (asegúrate de que la cámara esté correctamente configurada)
video_capture = cv2.VideoCapture('http://192.168.1.87:4747/video')  # 0 para la cámara predeterminada

while True:
    # Capturar un solo fotograma de video
    ret, frame = video_capture.read()
    if not ret:
        break
    
    # Convertir la imagen de BGR (OpenCV) a RGB (face_recognition)
    rgb_frame = frame[:, :, ::-1]

    # Encontrar todas las caras y codificaciones faciales en el fotograma de video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop a través de cada cara en este fotograma de video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Predecir la identidad de la persona utilizando SVM
        face_encoding = face_encoding.reshape(1, -1)
        result = model.predict(face_encoding)
        prob = model.decision_function(face_encoding).max()

        name = "Desconocido"
        
        if prob > 0.5:
            name = persons[result[0]]
        
        # Dibujar un rectángulo alrededor de la cara
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Escribir una etiqueta con el nombre debajo de la cara
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Mostrar la imagen resultante
    cv2.imshow('Video', frame)

    # Presionar 'q' en el teclado para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el controlador de la cámara
video_capture.release()
cv2.destroyAllWindows()