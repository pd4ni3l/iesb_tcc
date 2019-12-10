import cv2

classificadorGatos = q('cascades/haarcascade_frontalcatface.xml')

imagem = cv2.imread('outros/gato3.jpg')

imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

gatosDetectados = classificadorGatos.detectMultiScale(imagemCinza, scaleFactor=1.02)

for (x, y, l, a) in gatosDetectados:
    imagem = cv2.rectangle(imagem, (x,y), (x + l, y + a), (0, 0, 254), 2)

cv2.imshow('Gatooooossssss!!', imagem)
cv2.waitKey()