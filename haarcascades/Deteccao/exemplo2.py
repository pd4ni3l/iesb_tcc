import cv2

classificadorFace - cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

classificadorOlhos = cv2.CascadeClassifier('cascades/haarcacade_eye.xml')

imagem = cv2.imread('pessoas/faceolho.jpg')
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)