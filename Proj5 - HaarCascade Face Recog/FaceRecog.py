# Game of Thrones Characters 

import cv2 as cv
import pafy as pf
import pickle

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

try :
#    eyeCascade= cv.CascadeClassifier('haarcascade_eye.xml')
    faceCascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
#    faceCascade = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.read("recognizers/face-trainner.yml")
    
#    url = "https://www.youtube.com/watch?v=aB_UvmPqXTg"
#    url = "https://www.youtube.com/watch?v=81Uz1mF09O4" #GOT Jamie and Cersei Lannister 
#    url = "https://www.youtube.com/watch?v=pFk2t3E2aWE" 
#    url = "https://www.youtube.com/watch?v=9QbltzIUV6w" # Avengers 
    url = "https://www.youtube.com/watch?v=VetyHT-rZx0" #GOT white walker Intro

    videoPafy = pf.new(url)
    best = videoPafy.getbest(preftype="webm")
    print (videoPafy.title)
    input_video = cv.VideoCapture(best.url)
    
#    start_frame_number = 60
#    input_video.set(cv.CAP_PROP_FPS, 60)
#    input_video.set(cv.CAP_PROP_POS_FRAMES, start_frame_number)
    
    length = int(input_video.get(cv.CAP_PROP_FRAME_COUNT))
    frame_number = 0
    while input_video.isOpened() :
        ret, frame = input_video.read()
        frame_number += 1
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray
                                             , scaleFactor=1.3
                                             , minNeighbors=6
                                             , minSize=(30, 30)
                                             , flags=cv.CASCADE_SCALE_IMAGE)
        
        print('Found {} faces!'.format(len(faces)))

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            face_detect = cv.rectangle(gray, (x, y), (x+w, y+h), (255,0,0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
#            eyes = eyeCascade.detectMultiScale(roi, scaleFactor=1.2, minNeighbors=5)      
#            for (ex,ey,ew,eh) in eyes:
#                cv.rectangle(roi,(ex,ey),(ex+ew,ey+eh), 255, 2)
            id_,conf = recognizer.predict(roi_gray)
            if conf>=4 and conf <= 85:
                font = cv.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color = (255, 255, 255)
                stroke = 2
                cv.putText(frame, name, (x,y), font, 1, color, stroke, cv.LINE_AA)
            color = (255, 0, 0) #BGR 0-255 
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
                
        cv.imshow ('frame', frame)

        if not ret:
            break
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
    input_video.release()
    cv.destroyAllWindows()
except KeyboardInterrupt:
    raise
    print ('Interrupted')
except Exception as e:
    print(e)
    input_video.release()
    cv.destroyAllWindows()
    