import os
import dlib
import numpy as np
from sklearn import svm
import pickle
import face_recognition

# Ruta donde hayas almacenado Data
dataPath = '/content/Data'
modelPath = '/content/Modelos'

if not os.path.exists(modelPath):
    os.makedirs(modelPath)

peopleList = os.listdir(dataPath)
print('Lista de personas:', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = os.path.join(dataPath, nameDir)
    print('Leyendo las imágenes')

    for fileName in os.listdir(personPath):
        print('Rostros:', nameDir + '/' + fileName)
        image_path = os.path.join(personPath, fileName)
        image = face_recognition.load_image_file(image_path)
        face_bounding_boxes = face_recognition.face_locations(image)

        if len(face_bounding_boxes) != 1:
            # Si la imagen no contiene un solo rostro, se ignora.
            print(f"Error: {image_path} no contiene exactamente un rostro.")
            continue
        
        # Extrae los encodings (vectores de características) de la imagen
        face_enc = face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0]
        facesData.append(face_enc)
        labels.append(label)

    label += 1

print(f"Total de imágenes cargadas: {len(facesData)}")

# Variable para seleccionar el tipo de modelo
model_type = 'SVM'  # En este ejemplo usamos SVM

# Inicializar el modelo basado en la variable model_type
if model_type == 'SVM':
    model = svm.SVC(gamma='scale')
else:
    raise ValueError("Tipo de modelo no reconocido. Usa 'SVM'.")

# Entrenando el reconocedor de rostros
print("Entrenando...")
model.fit(facesData, labels)
print("Entrenamiento completado")

# Almacenando el modelo obtenido
model_filename = 'modeloSVM.pkl'
model_filepath = os.path.join(modelPath, model_filename)
with open(model_filepath, 'wb') as model_file:
    pickle.dump(model, model_file)
print(f"Modelo almacenado en {model_filepath}")
