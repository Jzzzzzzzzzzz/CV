import cv2
import mediapipe as mp
import time
from imageai.Detection import ObjectDetection

# https://www.youtube.com/watch?v=f7IZBmjbpqU&t=639s&ab_channel=BeTry%7C%D0%9F%D1%80%D0%BE%D0%B3%D1%80%D0%B0%D0%BC%D0%BC%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5

capture = cv2.VideoCapture(0)
draw = mp.solutions.drawing_utils

mp_holistic = mp.solutions.holistic
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath("model/retinanet_resnet50_fpn_coco-eeacb38b.pth")
detector.loadModel()
output_path = "image_web-video.jpg"

array_detection = []
count = 0


# def phone_obj_dec(func):
#     def wrapper(*args,**kwargs):
#         start = time.time()
#         result = func(*args,**kwargs)
#         finish = time.time()
#         return result
#     print(finish-start)
#     return wrapper
#
#     return start
# @phone_obj_dec
start_phone = 0
list = []
def phone_obj(obje):
    global count
    print(count)
    if obje == "cell phone":
        start_phone = time.time()
        list.append(start_phone)
        print(start_phone)
        count += 1
        if count >= 4:
            print("hello, Apple Samsung phone and all brands")
            finish_phone  = time.time()
            print(finish_phone-list[0])

            # return print(f"{finish-start} секунд прошло,человек не заинтересован(ps в дальнейшем будем просто отнимать всё от 100% и тд) ")
            print(f"{int(finish_phone-list[0])}The person is not interested, he is on the phone")


finish = 0
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while capture.isOpened():
        ret, frame = capture.read()

        start = time.time()
        # print(start)
        if start - finish > 2:
            detections = detector.detectObjectsFromImage(input_image=frame, output_image_path=output_path,
                                                         minimum_percentage_probability=10)
            finish = time.time()

            for i in detections:
                print(i["name"])
                phone_obj(i["name"])

        # frame = cv2.flip(frame, 1)
        # recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # frame image
        # Make detections
        results = holistic.process(image)
        # face_landmarks, pose_landmark, mleft_landmarks, right_landmarks
        # print(results.face_landmarks)
        # Recolor image back to  BGE for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # frame image
        # 1. Draw face landmarks
        draw.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                            draw.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                            draw.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

        # 2. Right Hand
        draw.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                            draw.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

        # 3. Left Hand
        draw.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                            draw.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
        # 4.Pose Detections
        # draw.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        #     draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
        #     draw.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

        cv2.imshow("Webcam Feed", image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

capture.release()
cv2.destroyAllWindows()

