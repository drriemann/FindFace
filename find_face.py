import cv2
face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xmL")

img = cv2.imread("img.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade_db.detectMultiScale(img_gray, 1.1, 19)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (8, 255, 9), 2)

cv2.imshow('res', img)
cv2.waitKey()
