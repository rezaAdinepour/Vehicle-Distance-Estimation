# import libraries
import cvzone
from ultralytics import YOLO
import cv2
import math


# calculate TTC index
def Culculate_ttc(carspeed, dist):
    global TTC
    relative_speed = round(math.sqrt((carspeed * carspeed) + 2 * acceleration * dictans), 2)
    TTC = round((relative_speed / dist), 2)



# load YOLO.V8 pretrained weight
model = YOLO('Yolo.Weights/yolov8n.pt')

# labels
coco_file = 'coco.names'
classes = []

Car_speed = 90.0
acceleration = 1
TTC = 0


with open(coco_file, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')



cap = cv2.VideoCapture('video.mp4')
while True:

    ret, img =  cap.read()
    resultse = model(img, stream=True)

    for r in resultse:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)

            cvzone.cornerRect(img, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classes[cls]


            focal = 3017.142857142857
            wights_in_image = w
            real_wights = 3.2
            real_wights_small = 0.45



            if(currentClass == 'car' or currentClass =='bus' or currentClass == 'turk'):
                dictans = round((real_wights * focal) / wights_in_image, 2)
                Culculate_ttc(Car_speed, dictans)
                if(TTC <= 2 and TTC >= 1.5):
                    cvzone.putTextRect(img, f' {classes[cls]} Denger!!', (max(0, x1), max(35, y1)),
                                       scale=2, thickness=3, offset=10)
                else:
                    cvzone.putTextRect(img, f' {classes[cls]} {dictans} TTC value', (max(0, x1), max(35, y1)),
                                    scale=2, thickness=3, offset=10)


            if(currentClass == 'bicycle' or currentClass == 'motobike' or currentClass == 'person'):
                dictans = round((real_wights_small * focal) / wights_in_image, 2)
                Culculate_ttc(Car_speed, dictans)
                if (TTC <= 2 and TTC >= 1.5):
                    cvzone.putTextRect(img, f' {classes[cls]} Danger!!!', (max(0, x1), max(35, y1)),
                                    scale=1, thickness=3, offset=10)
                else:
                    cvzone.putTextRect(img, f' {classes[cls]} {TTC} Value', (max(0, x1), max(35, y1)),
                                       scale=1, thickness=3, offset=10)




    cv2.imshow('Cars', img)
    cv2.waitKey(1)