from imageai.Detection import ObjectDetection

detector = ObjectDetection()

detector.setModelTypeAsRetinaNet()

detector.setModelPath("model/retinanet_resnet50_fpn_coco-eeacb38b.pth")

detector.loadModel()

input_path = "../Get_object_phone/new_Elonn.jpg"
output_path = "../Get_object_phone/new_Elonns.jpg"


detections = detector.detectObjectsFromImage(input_image=input_path,output_image_path=output_path,minimum_percentage_probability=10)

for i in detections:
    print(f" Object:{ i['name'] } { i['percentage_probability'] }%   { i['box_points'] }")