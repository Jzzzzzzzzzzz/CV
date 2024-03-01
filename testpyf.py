import cv2

import mediapipe as mp
import time
capture  = cv2.VideoCapture(0)
capture2  = cv2.VideoCapture(1)



mp_face_mesh = mp.solutions.face_mesh
draw = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles 



capture.set(cv2.CAP_PROP_FRAME_WIDTH,100)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 100)
while True:
    time.sleep(0.1)
    _,img = capture.read()
   # applying face mesh model using MediaPipe
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    face_mesh_results = mp_face_mesh.FaceMesh(refine_landmarks = True).process(img)
    # draw annotations on the image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    

    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
                 
            
                 draw.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )
    cv2.imshow("camera",img)

    if  cv2.waitKey(1) == ord("q"):
        break