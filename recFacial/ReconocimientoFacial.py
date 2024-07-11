import cv2
import requests
import threading
import time
import sys

dataPath = './Data'
persons = ['Copito', 'Nacho', 'Nico V', 'Roby', 'Seba H']
print('imagePaths=', persons)

# Inicializando la captura de video
#cap = cv2.VideoCapture('http://192.168.1.86:4747/video')
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error al abrir la cámara.")
    sys.exit(1)

recognizer = cv2.face.LBPHFaceRecognizer_create()

# Leyendo el modelo
recognizer.read('./Modelos/modeloLBPHFace.xml')
print("Modelo cargado.")

# Inicializando el clasificador de rostros
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
url = 'http://localhost:8000/asistencia/'
client = requests.session()

# Obteniendo CSRF token
try:
    response = client.get(url)
    if 'csrftoken' in client.cookies:
        csrftoken = client.cookies['csrftoken']
        print("Token encontrado:", csrftoken)
    else:
        csrftoken = None
        print("Token no encontrado.")
except requests.RequestException as e:
    print("Error al conectar con el servidor:", e)
    csrftoken = None

# Variable para controlar el estado de detección por persona
detected_persons = {}

# Función para procesar y enviar datos en un hilo
def procesar_y_enviar_datos(frame, persons, recognizer, faceClassif, url, csrftoken):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        auxFrame = gray.copy()
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = recognizer.predict(rostro)
        
        cv2.putText(frame, '{}'.format(result), (x, y-5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
        
        if result[1] < 80:
            person_name = persons[result[0]]
            cv2.putText(frame, '{}'.format(person_name), (x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Verificar si ya se detectó a esta persona recientemente
            if person_name not in detected_persons:
                detected_persons[person_name] = time.time()  # Registrar el tiempo de detección
                
                if csrftoken:
                    try:
                        print(result[0])
                        print(result[1])
                        myobj = dict(person=person_name, csrfmiddlewaretoken=csrftoken)
                        x = client.post(url, data=myobj, headers=dict(Referer=url))
                        if x.status_code == 200:
                            print("Datos enviados correctamente.")
                        else:
                            print("Error al enviar datos:", x.status_code)
                    except requests.RequestException as e:
                        print("Error al enviar datos al servidor:", e)
                else:
                    print("No se puede enviar datos porque el token CSRF no está disponible.")
                print(person_name)

# Bucle principal de captura
frame_count = 0
while True:
    # Capturar fotograma
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar la imagen.")
        break
    frame_count += 1
    
    # Procesar el fotograma actual en un hilo
    threading.Thread(target=procesar_y_enviar_datos, args=(frame.copy(), persons, recognizer, faceClassif, url, csrftoken)).start()
    
    # Eliminar personas que no están en el cuadro actual después de 5 segundos
    current_time = time.time()
    to_remove = [person_name for person_name, detection_time in detected_persons.items() if current_time - detection_time > 5]
    for person_name in to_remove:
        del detected_persons[person_name]
    
    # Mostrar el fotograma original en el hilo principal
    cv2.imshow('frame', frame)
    cv2.waitKey(1)  # Esperar un breve periodo para mostrar el fotograma

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
