import cv2 
import mediapipe as mp

# face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# while True:
#     _,img = capture.read()
#     results = hands.process(img)
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade_db.detectMultiScale(img_gray, 1.1, 19)
# for (x,y,w,h) in faces:
#     cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2) 
# denormalize_coordinates = draw._normalized_to_pixel_coordinates
capture2  = cv2.VideoCapture(0)
capture  = cv2.VideoCapture("output.avi")
face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avii', fourcc, 20.0, (640, 480))
mp_facemesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles 

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
draw = mp.solutions.drawing_utils 
denormalize_coordinates = draw._normalized_to_pixel_coordinates

while True:
    _,img = capture.read()
    results = hands.process(img)
    out.write(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade_db.detectMultiScale(img_gray, 1.1, 19)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2) 
    if results.multi_hand_landmarks :
        for hand in results.multi_hand_landmarks:
            h,w,color = img.shape 
            draw.draw_landmarks(img,hand,mp_hands.HAND_CONNECTIONS)
            x8,y8 = int(hand.landmark[8].x * w), int(hand.landmark[8].y *h)
            print(x8,y8)
            cv2.circle(img,(x8,y8),15, (0,255,0),cv2.FILLED)
    cv2.imshow("camera",img)

    if  cv2.waitKey(1) == ord("q"):
        break

capture.release()
out.release()
cv2.destroyAllWindows()