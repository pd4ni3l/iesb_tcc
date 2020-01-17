# Este código se baseia no código do Prof. Jones Granatyr do curso Reconhecimento de faces e de objetos com Python e Dlib
# https://www.udemy.com/course/reconhecimento-de-faces-e-de-objetos-com-python-e-dlib/
# Importação dos módulos
import cv2
import dlib
import os # acessa o file system
import glob # percorre a pasta

# Definições de detector de face e detector de pontos faciais
detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
# Indicar a rede CNN a ser utilizada
reconheciemntoFacial = dlib.face_recognition_model_v1("recursos/dlib_face_recognition_resnet_model_v1.dat")

# Leitura da pasta com as imagens
for arquivo in glob.glob(os.path.join("dataset/*", "*.png")):
    imagem = cv2.imread(arquivo)
    # Detectar se existe faces na imagem
    # Quando as imagens não são pequenas não é necessário outros parametros como escala para melhorar a detecção
    facesDetectadas = detectorFace(imagem, 1)
    numeroFacesDetectadas = len(facesDetectadas)
    # Sanitizando
    # Verificar se existe face na imagem
    if numeroFacesDetectadas > 1:
        # Avisar que existe mais de uma face e sair informando o arquivo
        print("Número de faces detectadas na imagem: {} arquivo {}".format(numeroFacesDetectadas, arquivo))
        exit(0) # Aborta a varredura da pasta para correção
    elif numeroFacesDetectadas < 1:
        # Avisar com que não tem imagem e sair informando o arquivo
        print("Nenhuma face detectadas no arquivo {}".format(arquivo))
        exit(0) # Aborta a varredura da pasta para correção