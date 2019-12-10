import cv2

classificador = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')


imagem = cv2.imread('pessoas/pessoas4.jpg')
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# cv2.imshow('Beatles', imagem)
# cv2.waitKey()
# facesDetectadas = classificador.detectMultiScale(imagem)
# Melhorando a detecção utilizando o scaleFactor. Quanto menor o valor mais lento é. Valor default 1.1
# facesDetectadas = classificador.detectMultiScale(imagem, scaleFactor=1.09)
# Melhorando a quantidade de vizinhos. Valores altos menos detecções. minNeighbors
# facesDetectadas = classificador.detectMultiScale(imagem, scaleFactor=1.09, minNeighbors=7)
# minSize especifica o tamanho mínimo do tamanho das faces.
# o código a baixo funcionou para pessoas2, faceolho, pessoas1, beatles
# facesDetectadas = classificador.detectMultiScale(imagem, scaleFactor=1.1, minNeighbors=10, minSize=(30,30))

facesDetectadas = classificador.detectMultiScale(imagem, scaleFactor=1.1, minNeighbors=10, minSize=(30,30))
print(len(facesDetectadas))
print(facesDetectadas)

# Desenhando os quadrados
for x, y, l, a in facesDetectadas:
    print(x, y, l, a)
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)

cv2.imshow('Pessoas', imagem)
cv2.waitKey()