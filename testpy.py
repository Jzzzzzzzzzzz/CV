import cv2 
import mediapipe as mp
import time as tm

capture  = cv2.VideoCapture(0)
capture2  = cv2.VideoCapture("output.avi")
capture.set(10, 100)  # Brightness
face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avii', fourcc, 20.0, (640, 480))



mp_hands = mp.solutions.hands
hands = mp_hands.Hands(False)
draw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

denormalize_coordinates = draw._normalized_to_pixel_coordinates

while True:
    _,img = capture.read()
    img = cv2.flip(img, 1)
    out.write(img)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = hands.process(img)
    faces = face_cascade_db.detectMultiScale(img_gray, 1.1, 19)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2) 
    if results.multi_hand_landmarks :
        for hand in results.multi_hand_landmarks:
            h,w,c = img.shape
            draw.draw_landmarks(img, hand,mp_hands.HAND_CONNECTIONS)
            x8,y8 = int(hand.landmark[8].x * w), int(hand.landmark[8].y *h)
            print(x8,y8)
            cv2.circle(img, (x8, y8), 10, (0, 255, 0 ), cv2.FILLED)
    cTime = tm.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow("camera", img)
    if  cv2.waitKey(1) == ord("q"):
        break




capture.release()
out.release()
cv2.destroyAllWindows()