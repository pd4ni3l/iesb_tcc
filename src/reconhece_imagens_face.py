# Este código se baseia no código do Prof. Jones Granatyr do curso Reconhecimento de faces e de objetos com Python e Dlib
# https://www.udemy.com/course/reconhecimento-de-faces-e-de-objetos-com-python-e-dlib/
# Importação dos módulos
import cv2
import dlib
import os # acessa o file system
import glob # percorre a pasta
import numpy as np
import pickle as cPickle

# Definições de detector de face e detector de pontos faciais
detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
# Indicar a rede CNN para ser utilizada
reconheciemntoFacial = dlib.face_recognition_model_v1("recursos/dlib_face_recognition_resnet_model_v1.dat")
# Carregar os indices criados previamente
# Windows
indices = cPickle.loads(open("recursos/indices_familia.pickle",'rb').read())
# Mac
# indices = np.load("recursos/indices_familia.pickle")
# Carregar descritores faciais criados anteriormente
descritoresFaciais = np.load("recursos/descritores_familia.npy")
# Definir o limiar que o KNN usará para se posicionar no vetor de características
# Se limiar for igual a zero (0) quer dizer que a imagem do rosto tem que ser exatamente igual a da base do descritor
limiar = 0.5

# Varrer pasta com imagens que nao foram utilizadas para criar a base de treinamento
for arquivo in glob.glob(os.path.join("dataset/*", "*.jpg")):
    imagem = cv2.imread(arquivo)
    facesDetectadas = detectorFace(imagem, 1)
    for face in facesDetectadas:
        e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
        # Pegar os pontos faciais da imagem atual
        pontosFaciais = detectorPontos(imagem, face)
        # Gerar descritor facial da imagem atual
        descritorFacial = reconheciemntoFacial.compute_face_descriptor(imagem, pontosFaciais)
        # Gerar lista como feito no gera_descritor_face.py para fazer a comparação
        listaDescritorFacial = [fd for fd in descritorFacial]
        # Gerar array numpy com os valores coletados da lista
        npArrayDescritorFacial = np.asarray(listaDescritorFacial, dtype=np.float64)
        # Adicionar a coluna ao array
        npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]
        # Comparar npArrayDescritorFacial com descritoresFaciais utilizando KNN
        # Calcular a distancia Euclidiana. Normalizar os dados
        distancias = np.linalg.norm(npArrayDescritorFacial - descritoresFaciais, axis=1)
        # Visualizar distancias
        # print("Distancias: {}".format(distancias))
        '''
        Distancias: [0.50116532 0.47273572 0.5515244  0.45538878 0.4340941  0.56476209
                    0.42094817 0.29000221 0.45459123 0.49568262 0.3852318  0.524574
                    0.50753166 0.69505442 0.78924636 0.74964648 0.6998796  0.90604675
                    0.90511255 0.90065656 0.88459135 0.87015046 0.89418785 0.87984072
                    0.88287158 0.95849524 0.97739964 0.91053409 0.92449609 0.89419064
                    0.77339664 0.70950361 0.69738734]
 Explicando:
    Temos 33 imagens com 128 características, pegou a primeira imagem e comparou com todas as 33
    Agora temos que pegar a menor distância, para informar no bound box que o rosto detectado na imagem autal
    pode ser aquele com a menor distancia.
    No caso 0.29000221 é a menor distancia
    print("Menor distancia:  ", min(distancias))
        '''
        # Visualizando menor distancia e arquivo analisado
        # print("Menor distancia: {}. Arquivo: {}".format(min(distancias), arquivo))
        # minimo recebe a posição do menor valor
        minimo = np.argmin(distancias)
        distanciaMinima = distancias[minimo]
        # Visualizando
        # print(distanciaMinima)
        # print(indices)
        # exit(0)
        # Controlar verificando o limiar
        if distanciaMinima <= limiar:
            # Pegar o nome dessa pessoa no indice e somente uma (1) posiçao. Fatiar a string para pegar o nome.
            # Windows
            nome = os.path.split(indices[minimo])[0].split("\\")[1]
            # Mac
            # nome = os.path.split(indices[minimo])[0].split("/")[1]
            # print(nome)
        else:
            nome = 'Desconhecido'

        # Desenhar bounding boxes
        cv2.rectangle(imagem, (e, t), (d, b), (0,255,255), 2)
        # Adicionar texto ao bounding box
        texto = "{} {:.4f}".format(nome, distanciaMinima) # Formatação de qual a distancia que foi encontrada.
        cv2.putText(imagem, texto, (d, t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,255,255))

    cv2.imshow("Detectando com HOG", imagem)
    cv2.waitKey(0)

# Destroi todas as janelas
cv2.destroyAllWindows()