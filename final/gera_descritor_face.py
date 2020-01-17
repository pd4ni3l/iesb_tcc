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
# Indicar a rede CNN a ser utilizada
reconheciemntoFacial = dlib.face_recognition_model_v1("recursos/dlib_face_recognition_resnet_model_v1.dat")
# Variáveis de controle
indice = {}
idx = 0
descritoresFaciais = None

# Leitura da pasta com as imagens
for arquivo in glob.glob(os.path.join("dataset/*", "*.png")):
    imagem = cv2.imread(arquivo)
    # Detectar se existe faces na imagem
    # Quando as imagens não são pequenas não é necessário outros parametros como escala para melhorar a detecção
    facesDetectadas = detectorFace(imagem, 1)
    numeroFacesDetectadas = len(facesDetectadas)
    # Sanitizando
    # Verifica se existe face na imagem
    if numeroFacesDetectadas > 1:
        # Avisar que existe mais de uma face e sair informando nome do arquivo
        print("Número de faces detectadas na imagem: {} arquivo {}".format(numeroFacesDetectadas, arquivo))
        exit(0) # Aborta a varredura da pasta para correção
    elif numeroFacesDetectadas < 1:
        # Avisar que não tem imagem e sair informando o nome do arquivo
        print("Nenhuma face detectadas no arquivo {}".format(arquivo))
        exit(0) # Aborta a varredura da pasta para correção

    # Identificar os 68 pontos faciais
    for face in facesDetectadas:
        # O parâmetro imagem é a imagem toda e face é somente o pedaço que contém o rosto
        pontosFaciais = detectorPontos(imagem, face)
        descritorFacial = reconheciemntoFacial.compute_face_descriptor(imagem, pontosFaciais)
        # O resultado do descritorFacial é um vetor com 128 posições que descrevem a face encontrada
        # Visualizar as informações encontradas
        # print(format(arquivo))
        # print(len(descritorFacial))
        # print(descritorFacial)
        # Colocar em uma lista os dados do Dlib descritorFacial para uma lista de tamanho 128
        listaDescritorFacial = [df for df in descritorFacial]
        # Visualizar as informações
        # print(listaDescritorFacial)
        # Converter a lista em um vetor numpy
        npArrayDescritorFacial = np.asarray(listaDescritorFacial, dtype=np.float64)
        # Visualizar as informações
        # print(npArrayDescritorFacial)
        # Alterando o array para 1x128
        # Adiciona uma linha contendo o descritor de forma
        npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]
        # print(npArrayDescritorFacial)
        # Concatenação
        if descritoresFaciais is None:
            descritoresFaciais = npArrayDescritorFacial
        else:
            descritoresFaciais = np.concatenate((descritoresFaciais, npArrayDescritorFacial), axis=0)

        indice[idx] = arquivo
        idx += 1
    # Verificar o funcionamento
    # cv2.imshow("Validacao", imagem)
    # cv2.waitKey(0)

# Visualizar informações da concatenação
"""
print("Tamanho (qtd imagens): {}, formato do array {}, arquivo {}"\
    .format(len(descritoresFaciais), descritoresFaciais.shape, indice))
"""
# Salvar o array com as caracteristicas das faces
np.save("recursos/descritores_familia.npy", descritoresFaciais)
# Salvar o classificador
with open("recursos/indices_familia.pickle", 'wb') as f:
    cPickle.dump(indice, f)

# Limpar todas as janelas
# cv2.destroyAllWindows()
