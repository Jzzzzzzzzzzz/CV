# face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# while True:
#     _,img = capture.read()
#     results = hands.process(img)
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade_db.detectMultiScale(img_gray, 1.1, 19)
# for (x,y,w,h) in faces:
#     cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)
# denormalize_coordinates = draw._normalized_to_pixel_coordinates