# Importação de pacotes
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2

# Definindo arquivos haarcascade e faces codificadas (pickle)
cascade = "recursos/haarcascade_frontalface_default.xml"
encodings = "recursos/faces_familia_codificada.pickle"

# Carregando arquivo de faces conhecidas e arquivo de Haar de faces
print("[INFO] Carregando arquivo de faces conhecidas e arquivo Haar detector de faces...")
data = pickle.loads(open(encodings, "rb").read())
detector = cv2.CascadeClassifier(cascade)

# Inicializando o stream de video e aguardando 2 segundo de aquecimento do sensor
print("[INFO] Iniciando video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Inicializando o contador FPS
fps = FPS().start()

# loop sobre os frames de video do arquivo de stream
while True:
    # Redimencioando para 500px o stream de video para acelerar o processamento
    frame = vs.read()
    frame = imutils.resize(frame, width=500)

    # Converter BGR para RGB para reconhecimento de face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar faces em tons de cinza no frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    # rects = detector.detectMultiScale(gray, scaleFactor=1.1, minSize=(10,10))

    # Ajustar os parametros retornados do OpenCV para os bounding boxes
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

    # Calcular as faces encontradas em cada bounding boxes utilizando módulo face_recognition
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # Loop sobre a face encontrada
    for encoding in encodings:
        # Tentar encontrar a face detectada com as faces conhecidas utilizando módulo face_recognition
        matches = face_recognition.compare_faces(data["encodings"],
                                                 encoding)
        name = "Desconhecido"

        # Verificando se face foi encontrada
        if True in matches:
            # Procurar os indices de todas as faces encontradas e inicializar
            # o dicionario de contagem total de quantas vezes cada face foi detectada
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # Determinar que a face com maior numero de "votos" é a face reconhecida
            # Selecionando a primeira da lista
            name = max(counts, key=counts.get)

        # Atualizar a lista de nomes
        names.append(name)

    # Loop sobre as faces para desenhar os bounding boxes
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # Escrever o nome previsto da face detectada
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)

    # Mostrar a imagem na tela
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Pressionar 'q' para parar o loop
    if key == ord("q"):
        break

    # Atualizar o contador FPS
    fps.update()

# Parar o timer a informar o FPS
fps.stop()
print("[INFO] Tempo decorrido: {:.2f}".format(fps.elapsed()))
print("[INFO] FPS aproximado: {:.2f}".format(fps.fps()))

# Destruir as telas
cv2.destroyAllWindows()
vs.stop()