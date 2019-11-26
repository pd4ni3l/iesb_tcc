import cv2

img = cv2.imread('examples/IMG_0070.jpeg', cv2.IMREAD_UNCHANGED)

print('Original Dimensions : ', img.shape)

scale_percent = 40  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

print('Resized Dimensions : ', resized.shape)

cv2.imshow("Resized image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# read image as grey scale
# grey_img = cv2.imread('/home/img/python.png', cv2.IMREAD_GRAYSCALE)

# save image
# status = cv2.imwrite('/home/img/python_grey.png', grey_img)

# print("Image written to file-system : ", status)