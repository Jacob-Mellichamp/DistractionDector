"""Human facial landmark detector based on Convolutional Neural Network."""
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import math
import time

import FaceDetection
import MarkerDector

#Previous Assignment Functions
def getCameraMatrix():
    K = np.loadtxt("cameraMatrix.csv", delimiter=",", usecols=range(3))
    return K



def getDistortionValues():
    distortion = np.loadtxt("distortion_coef.csv", delimiter=",", usecols=range(5))
    return distortion

 
def radian_to_degrees(rad):
    return float(rad * 57.2958)


#Helper function for main
#@Pre: takes two points p1, p2
#@Post: creates slope line and returns angle of line and returns arc tangent of the line
def calcAngle(p1, p2):
    ang = None
    try:
        m = (p1[1] - p2[1])/(p1[0] - p2[0])
        ang = int(math.degrees(math.atan(-1/m)))
    except:
        ang = 90
    return ang


#Helper functions for main
#@Pre: Face object discovered in While Loop
#@Post: Resize the face, and change color
def optimize_face(facebox, img):
    face_img = img[facebox[1]: facebox[3], facebox[0]: facebox[2]]
    face_img = cv2.resize(face_img, (128, 128))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    return face_img
    



def main(is_simple_approach):


    mark_detector = MarkerDector.MarkDetector()
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    size = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            ])

    #TODO: Define Camera Matrix
    camera_matrix = getCameraMatrix()

    #TODO: Define Distortion Coef
    dist_coeffs = getDistortionValues()



    #Storage of attention
    attention_accum = 0.0
    distracted_accum = 0.0


    ROT_X_WEIGHT = 165
    ROT_Y_WEIGHT = 0.0
    ROT_Z_WEIGHT = 40.0



    #try statement to Ctrl+C out of while loop
    try:
        # While Video Stream
        while True:
            starttime = time.time()

            #time.sleep(0.5)
            ret, img = cap.read()
            if ret == True:
                faceboxes = mark_detector.extract_cnn_facebox(img)

                for facebox in faceboxes:
                    face_img = optimize_face(facebox, img)

                    marks = mark_detector.detect_marks([face_img])
                    marks *= (facebox[2] - facebox[0])
                    marks[:, 0] += facebox[0]
                    marks[:, 1] += facebox[1]
                    shape = marks.astype(np.uint)

                    #Draw Markers on face
                    #mark_detector.draw_marks(img, marks, color=(0, 255, 0))
                    image_points = np.array([
                                            shape[30],     # Nose tip
                                            shape[8],     # Chin
                                            shape[36],     # Left eye left corner
                                            shape[45],     # Right eye right corne
                                            shape[48],     # Left Mouth corner
                                            shape[54]      # Right mouth corner
                                        ], dtype="double")

                    nose_points = np.array([
                        shape[27],
                        shape[28], 
                        shape[29],
                        shape[30]
                    ])
                    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)


                        
                    #Rotation Vector is in radians.... Convert to degrees
                    rot_x = "{0:.2f}".format(radian_to_degrees( rotation_vector[0]) + ROT_X_WEIGHT)
                    rot_y = "{0:.2f}".format(radian_to_degrees( rotation_vector[1]) + ROT_Y_WEIGHT)
                    rot_z = "{0:.2f}".format(radian_to_degrees( rotation_vector[2]) + ROT_Z_WEIGHT)
                    string_rot = "X: {0}".format( rot_x)
                    #cv2.putText(img, string_rot, (20,100), font, 2, (128, 255, 255), 3)
                    
                    
                    for p in image_points:
                        cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
                    
                    p1 = ( int(nose_points[0][0]), int(nose_points[0][1]))
                    p2 = ( int(nose_points[3][0]), int(nose_points[3][1]))

                    cv2.line(img, p1, p2, (255, 255, 0), 2)


                    if is_simple_approach:
                        ang = calcAngle(p1, p2)
                    
                        cv2.putText(img, str(ang), p1, font, 2, (0, 255, 0), 3)

                        if ((ang <= 10 and ang >= -10) or ang == 90):
                            attention_accum += (time.time()) - starttime
                            

                        else:
                            distracted_accum += (time.time()) - starttime

                    else:
                        #uncomment below lines to view the rotation matrix values
                        if ((float(rot_y) < 15 and (float(rot_y)) >= -15) and (float(rot_x) < 12 and float(rot_x) >= -12)):
                            attention_accum += (time.time()) - starttime
                            cv2.putText(img, rot_x, p1, font, 2, (0, 255, 0), 3)
                            #cv2.putText(img, rot_y, p1, font, 2, (0, 255, 0), 3)


                        else:
                            distracted_accum += (time.time()) - starttime
                            #cv2.putText(img, rot_y, p1, font, 2, (0, 0, 255), 3)
                            cv2.putText(img, rot_x, p1, font, 2, (0, 0, 255), 3)


                display_attention = "{0:.3f}".format(attention_accum)
                display_distracted = "{0:.3f}".format(distracted_accum)
                print("\n\n\n")
                print("Time Attentive: \t" + str(display_attention))
                print("Time Distracted: \t" + str(display_distracted))
                cv2.putText(img, str(display_attention), (20,100), font, 2, (0, 255, 0), 3)
                cv2.putText(img, str(display_distracted), (20, 300), font, 2, (0, 0, 255), 3)
                cv2.imshow("img", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        cap.release()

if __name__ == "__main__":
    use_arctan = True
    main(use_arctan)