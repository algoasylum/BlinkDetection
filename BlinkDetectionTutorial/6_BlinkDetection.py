#-----Step 1: Use VideoCapture in openCV-----
import cv2
import dlib
import math
BLINK_RATIO_THRESHOLD = 5.7

#-----Step 5: Getting to know blink ratio

def midpoint(point1 ,point2):
    return (point1.x + point2.x)/2,(point1.y + point2.y)/2

def euclidean_distance(point1 , point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_blink_ratio(eye_points, facial_landmarks):
    
    #loading all the required points
    corner_left  = (facial_landmarks.part(eye_points[0]).x, 
                    facial_landmarks.part(eye_points[0]).y)
    corner_right = (facial_landmarks.part(eye_points[3]).x, 
                    facial_landmarks.part(eye_points[3]).y)
    
    center_top    = midpoint(facial_landmarks.part(eye_points[1]), 
                             facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), 
                             facial_landmarks.part(eye_points[4]))

    #calculating distance
    horizontal_length = euclidean_distance(corner_left,corner_right)
    vertical_length = euclidean_distance(center_top,center_bottom)

    ratio = horizontal_length / vertical_length

    return ratio

#livestream from the webcam 
cap = cv2.VideoCapture(0)

'''in case of a video
cap = cv2.VideoCapture("__path_of_the_video__")'''

#name of the display window in openCV
cv2.namedWindow('BlinkDetector')

#-----Step 3: Face detection with dlib-----
detector = dlib.get_frontal_face_detector()

#-----Step 4: Detecting Eyes using landmarks in dlib-----
predictor = dlib.shape_predictor("BlinkDetection/shape_predictor_68_face_landmarks.dat")
#these landmarks are based on the image above 
left_eye_landmarks  = [36, 37, 38, 39, 40, 41]
right_eye_landmarks = [42, 43, 44, 45, 46, 47]

while True:
    #capturing frame
    retval, frame = cap.read()

    #exit the application if frame not found
    if not retval:
        print("Can't receive frame (stream end?). Exiting ...")
        break 

    #-----Step 2: converting image to grayscale-----
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #-----Step 3: Face detection with dlib-----
    #detecting faces in the frame 
    faces,_,_ = detector.run(image = frame, upsample_num_times = 0, adjust_threshold = 0.0)

    #-----Step 4: Detecting Eyes using landmarks in dlib-----
    for face in faces:
        
        landmarks = predictor(frame, face)

        #-----Step 5: Calculating blink ratio for one eye-----
        left_eye_ratio  = get_blink_ratio(left_eye_landmarks, landmarks)
        right_eye_ratio = get_blink_ratio(right_eye_landmarks, landmarks)
        blink_ratio     = (left_eye_ratio + right_eye_ratio) / 2

        if blink_ratio > BLINK_RATIO_THRESHOLD:
            #Blink detected! Do Something!
            cv2.putText(frame,"BLINKING",(10,50), cv2.FONT_HERSHEY_SIMPLEX,
            2,(255,255,255),2,cv2.LINE_AA)


    cv2.imshow('BlinkDetector', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

#releasing the VideoCapture object
cap.release()
cv2.destroyAllWindows()

