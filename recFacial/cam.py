import cv2
import pickle
import numpy as np

# Cargar el modelo SVM entrenado
with open('.\Modelos\modeloSVM.pkl', 'rb') as f:
    svm_model = pickle.load(f)

# Inicialización del clasificador Haar Cascade para la detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicialización de la cámara
cap = cv2.VideoCapture('http://192.168.1.86:4747/video')

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convertir a escala de grises para la detección de rostros
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostros en la imagen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Procesar cada rostro detectado
    for (x, y, w, h) in faces:
        # Dibujar un rectángulo alrededor del rostro detectado (opcional)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Recortar la región de interés (ROI)
        roi = gray[y:y+h, x:x+w]
        
        # Redimensionar la ROI al tamaño esperado por el modelo SVM
        roi_resized = cv2.resize(roi, (128, 128))
        
        # Convertir la imagen a un vector de características y aplanarla
        features = roi_resized.flatten()
        
        # Asegurar que el vector de características tenga exactamente 128 elementos
        if len(features) != 128:
            continue  # O manejar esto según tus necesidades
        
        # Realizar la predicción con el modelo SVM
        prediction = svm_model.predict([features])[0]
        
        # Mostrar la etiqueta predicha
        cv2.putText(frame, str(prediction), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Mostrar el frame con los rostros detectados
    cv2.imshow('Frame', frame)
    
    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
