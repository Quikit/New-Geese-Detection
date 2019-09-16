import cv2 as cv
import numpy as np

goose_file = 'data/geese-cascade12.xml'
goose_cascade = cv.CascadeClassifier(goose_file)
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print('Camera not loaded')
        
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    geese = goose_cascade.detectMultiScale(frame, 
                                           scaleFactor=2, 
                                           minNeighbors=3,
                                           minSize=(30,30))
    
    for (x,y,w,h) in geese:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
    
    cv.imshow('Geese Detection', frame)
    
    if cv.waitKey(30) & 0xff == 27:
        break
    
cap.release()
cv.destroyAllWindows()
