import cv2

llanta_cascade = cv2.CascadeClassifier('cars.xml')
carros_cascade = cv2.CascadeClassifier('cars_2.xml')
haar_cascade = cv2.CascadeClassifier('haarcascade_car.xml')
cap = cv2.VideoCapture(0)

while 1:

    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    llantas = llanta_cascade.detectMultiScale(gray, 1.3, 5)
    carros_2 = carros_cascade.detectMultiScale(gray, 1.3, 5)
    haarcar = haar_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in llantas:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Llanta', (x - w, y - h), font, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
    for (x, y, w, h) in carros_2:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Llanta', (x - w, y - h), font, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
    for (x, y, w, h) in haarcar:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Llanta', (x - w, y - h), font, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()

cv2.destroyAllWindows()
