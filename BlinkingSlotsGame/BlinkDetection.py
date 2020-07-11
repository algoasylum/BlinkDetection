import cv2
import dlib
import math
import queue


close_it = False
q  = queue.Queue()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
left_eye_landmarks  = [36, 37, 38, 39, 40, 41]
right_eye_landmarks = [42, 43, 44, 45, 46, 47]
BLINK_RATIO_THRESHOLD = 5.7

class Blinking:
    
    def __init__(self ): 
        global q
        self.cap = cv2.VideoCapture(0)
        print('intialized Blink detection')       
    
    def midpoint(self,point1 ,point2):
        return (point1.x + point2.x)/2,(point1.y + point2.y)/2

    def euclidean_distance(self,point1 , point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def get_blink_ratio(self,eye_points, facial_landmarks):
        
        #loading all the required points
        corner_left  = (facial_landmarks.part(eye_points[0]).x, 
                        facial_landmarks.part(eye_points[0]).y)
        corner_right = (facial_landmarks.part(eye_points[3]).x, 
                        facial_landmarks.part(eye_points[3]).y)
        
        center_top    = self.midpoint(facial_landmarks.part(eye_points[1]), 
                                facial_landmarks.part(eye_points[2]))
        center_bottom = self.midpoint(facial_landmarks.part(eye_points[5]), 
                                facial_landmarks.part(eye_points[4]))

        #calculating distance
        horizontal_length = self.euclidean_distance(corner_left,corner_right)
        vertical_length = self.euclidean_distance(center_top,center_bottom)

        ratio = horizontal_length / vertical_length

        return ratio

    def start(self):
        
        # Buffer count between two blinks. This prevents detection of accidental blinks which 
        # occur involuntary after a voluntary blink. 
        buffer = 0

        #Keeps count of the number of times the subject blinked.
        blink_counter = 0

        #stop the application when 3 blinks are detected
        while blink_counter < 3:

            retval, frame = self.cap.read()

            if not retval:
                print("Can't receive frame (stream end?). Exiting ...")
                break 
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
           
            faces,_,_ = detector.run(image = frame, upsample_num_times = 0, adjust_threshold = 0.0)
            
            for face in faces:

                landmarks = predictor(frame, face)
               
                left_eye_ratio  = self.get_blink_ratio(left_eye_landmarks, landmarks)
                right_eye_ratio = self.get_blink_ratio(right_eye_landmarks, landmarks)
                blink_ratio     = (left_eye_ratio + right_eye_ratio) / 2
                
                if blink_ratio > BLINK_RATIO_THRESHOLD and buffer > 3:
                    buffer = 0
                    blink_counter+=1
                    q.put(blink_counter)
                    print("Blinked detected")
                
            buffer+=1

        self.cap.release()
        cv2.destroyAllWindows()
