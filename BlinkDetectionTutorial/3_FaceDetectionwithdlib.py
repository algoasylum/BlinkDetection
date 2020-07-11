#------Step 1: Use VideoCapture in openCV------
import cv2
import dlib

#livestream from the webcam 
cap = cv2.VideoCapture(0)

'''in case of a video
cap = cv2.VideoCapture("__path_of_the_video__")'''

#name of the display window in openCV
cv2.namedWindow('BlinkDetector')

#------Step 3: Face detection with dlib------
detector = dlib.get_frontal_face_detector()

while True:
    #capturing frame
    retval, frame = cap.read()

    #exit the application if frame not found
    if not retval:
        print("Can't receive frame (stream end?). Exiting ...")
        break 

    #------Step 2: converting image to grayscale------
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #------Step 3: Face detection with dlib----
    #detecting faces in the frame 
    faces,_,_ = detector.run(image = frame, upsample_num_times = 0,
                            adjust_threshold = 0.0)
    
    
    cv2.imshow('BlinkDetector', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

#releasing the VideoCapture object
cap.release()
cv2.destroyAllWindows()