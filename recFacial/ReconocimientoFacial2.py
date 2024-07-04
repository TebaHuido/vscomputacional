import cv2
import requests
import sys

dataPath = './Data'
persons = ['Nacho', 'Roby', 'Teba']
print('imagePaths=', persons)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Leyendo el modelo
face_recognizer.read('modeloLBPHFace2.xml')
print("Modelo cargado.")

# Inicializando la captura de video
cap = cv2.VideoCapture('http://192.168.1.84:4747/video')
if not cap.isOpened():
    print("Error al abrir la cámara.")
    sys.exit(1)

# Inicializando el clasificador de rostros
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

url = 'http://localhost:8000/test'
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

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar la imagen.")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)  # Aplicar detección de bordes
    
    auxFrame = edges.copy()  # Utilizar la imagen de bordes
    
    faces = faceClassif.detectMultiScale(edges, 1.3, 5)  # Utilizar edges en lugar de gray
    
    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)
        
        cv2.putText(frame, '{}'.format(result), (x, y-5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
        
        if result[1] < 80:
            cv2.putText(frame, '{}'.format(persons[result[0]]), (x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            if csrftoken:
                try:
                    myobj = dict(person=persons[result[0]], csrfmiddlewaretoken=csrftoken)
                    x = client.post(url, data=myobj, headers=dict(Referer=url))
                    if x.status_code == 200:
                        print("Datos enviados correctamente.")
                    else:
                        print("Error al enviar datos:", x.status_code)
                except requests.RequestException as e:
                    print("Error al enviar datos al servidor:", e)
            else:
                print("No se puede enviar datos porque el token CSRF no está disponible.")
            print(persons[result[0]])
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()