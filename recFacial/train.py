import cv2
import os
import numpy as np

dataPath = 'X:/Proyectos/proyectoVision/recFacial/Data' # Cambia a la ruta donde hayas almacenado Data
modelPath = 'X:/Proyectos/proyectoVision/recFacial/Modelos' # Ruta donde se almacenarán los modelos
if not os.path.exists(modelPath):
    os.makedirs(modelPath)

peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = os.path.join(dataPath, nameDir)
    print('Leyendo las imágenes')

    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '/' + fileName)
        image_path = os.path.join(personPath, fileName)
        image = cv2.imread(image_path, 0)
        if image is None:
            print(f"Error al cargar la imagen: {image_path}")
            continue
        labels.append(label)
        facesData.append(image)
        print(f"Imagen {image_path} cargada con tamaño: {image.shape}")
    label += 1

print(f"Total de imágenes cargadas: {len(facesData)}")
print(f"Dimensiones de la primera imagen: {facesData[0].shape}")

# Variable para seleccionar el tipo de modelo
model_type = 'Fisher'  # Cambia esto a 'Eigen' o 'Fisher' para usar otros modelos

# Inicializar el reconocedor basado en la variable model_type
if model_type == 'Eigen':
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    model_filename = 'modeloEigenFace.xml'
elif model_type == 'Fisher':
    face_recognizer = cv2.face.FisherFaceRecognizer_create()
    model_filename = 'modeloFisherFace.xml'
elif model_type == 'LBPH':
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    model_filename = 'modeloLBPHFace.xml'
else:
    raise ValueError("Tipo de modelo no reconocido. Usa 'Eigen', 'Fisher' o 'LBPH'.")

# Entrenando el reconocedor de rostros
print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))
print("Entrenamiento completado")

# Almacenando el modelo obtenido
model_filepath = os.path.join(modelPath, model_filename)
face_recognizer.write(model_filepath)
print(f"Modelo almacenado en {model_filepath}")
