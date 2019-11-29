import cv2
import dlib
import face_recognition
from PIL import Image
fonte = cv2.FONT_HERSHEY_COMPLEX_SMALL
#imagem = cv2.imread("fotos/grupo.0.jpg")
#imagem = cv2.imread("fotos/grupo.1.jpg")
#imagem = cv2.imread("fotos/grupo.2.jpg")
#imagem = cv2.imread("fotos/grupo.3.jpg")
#imagem = cv2.imread("fotos/grupo.4.jpg")
#imagem = cv2.imread("fotos/grupo.5.jpg")
#imagem = cv2.imread("fotos/grupo.6.jpg")
imagem = cv2.imread("fotos/grupo.7.jpg")

# # Haar
# detectorHaar = cv2.CascadeClassifier("recursos/haarcascade_frontalface_default.xml")
# imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
# facesDetectadasHaar = detectorHaar.detectMultiScale(imagemCinza, scaleFactor=1.1, minSize=(10,10))
# 
# # Hog
# detectorHog = dlib.get_frontal_face_detector()
# facesDetectadasHog = detectorHog(imagem, 2)
# 
# # CNN
# detectorCNN = dlib.cnn_face_detection_model_v1("recursos/mmod_human_face_detector.dat")
# 
# facesDetectadasCNN = detectorCNN(imagem, 2)

# face_recognition
#image = face_recognition.load_image_file("fotos/grupo.7.jpg") # transforma em vetor igual ao cv2.read
#print("face recognition load: ",image)
#print("cv2 imageread: ", imagem)

#facesDetectadasFR = face_recognition.face_locations(image,number_of_times_to_upsample=2, model="hog")
facesDetectadasFR = face_recognition.face_locations(imagem,model="hog")


# for (x, y, l, a) in facesDetectadasHaar:
#     cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 255, 0), 2)
#     cv2.putText(imagem, "Haar", (x, y - 5), fonte, 0.5, (0, 255, 0))
# 
# for face in facesDetectadasHog:
#     e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
#     cv2.rectangle(imagem, (e, t), (d, b), (0, 255, 255), 2)
#     cv2.putText(imagem, "Hog", (d, t), fonte, 0.5, (0, 255, 255))
# 
# for face in facesDetectadasCNN:
#     e, t, d, b, c = (int(face.rect.left()), int(face.rect.top()), int(face.rect.right()), int(face.rect.bottom()), face.confidence)
#     cv2.rectangle(imagem, (e, t), (d, b), (255, 255, 0), 2)
#     cv2.putText(imagem, "CNN", (d, t), fonte, 0.5, (255, 255, 0))
# 
for face_location in facesDetectadasFR:
    # Print the location of each face in this image
    top, right, bottom, left = face_location
    # desenhar os boxes
    # print(face_location) 
    cv2.rectangle(imagem, (left, top), (right, bottom),(174, 29, 196), 2)
    cv2.putText(imagem, "FR", (right, top), fonte, 0.5, (174, 29, 196, 0))  

    # You can access the actual face itself like this:
    # Separa as faces detectadas uma em cada janela
    # face_image = imagem[top:bottom, left:right]
    # pil_image = Image.fromarray(face_image)
    # pil_image.show()

cv2.imshow("Comparativo detectores", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
