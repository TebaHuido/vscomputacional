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
    
    # Convertir a escala de grises para la detección de rostros
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostros en la imagen
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Procesar cada rostro detectado
    for (x, y, w, h) in faces:
        # Recortar la región de interés (ROI)
        roi = gray[y:y+h, x:x+w]
        
        # Redimensionar la ROI al tamaño esperado por el modelo SVM
        roi_resized = cv2.resize(roi, (128, 128))  # Redimensionar a 128x128
        
        # Convertir la imagen a un vector de características
        features = roi_resized.flatten()[:128]  # Aplanar la imagen y tomar solo los primeros 128 elementos
        
        # Verificar las dimensiones de las características
        print(features.shape)  # Asegurarse de que tenga (128,) como dimensión
        
        # Realizar la predicción con el modelo SVM
        prediction = svm_model.predict([features])[0]
        
        # Dibujar un rectángulo alrededor del rostro detectado
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
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
