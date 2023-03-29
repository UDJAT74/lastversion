from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator
import cv2 as cv
from send_emails import send_emails
import threading
from playsound import playsound


model = YOLO("yolov8n.pt")

camera = cv.VideoCapture(0)

def generate_frames():
    while True:
        res, frame = camera.read()
        if not res:
            break
        else:
            result = model.predict(frame)
            for r in result:
                annotator = Annotator(frame)
                boxes=r.boxes
                for box in boxes:
                    b=box.xyxy[0]
                    c=box.cls
                    annotator.box_label(b,model.names[int(c)],3)
                    # print( int(b[0]),int(c) ) #return class name , c class id
                    # x, y, w, h= int(b[0]),int(b[1]),int(b[2]),int(b[3])
                    if int(c)==43:
                        cv.imwrite("screenShot.jpg", frame)
                        # thread = threading.Thread(target=playsound, args=("alarm3.wav",))
                        # thread.start()
                        # thread2 =threading.Thread(target=send_emails())
                        # thread2.start()
                    if int(c)==0  :
                        x, y, w, h = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                        thread3 = threading.Thread(target=tracking(frame,x, y, w, h))
                        thread3.start()
                        # tracking(frame,x, y, w, h)

            frame=annotator.result()
            # cv.imshow('YOLO V8 Detection', frame)
            # if cv.waitKey(1) & 0xFF == ord('q'):
            #     break

    # camera.release()
    # cv.destroyAllWindows()
            ret, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def tracking(frame,x, y, w, h):
    # img = cv.imread(r"D:\4th Year\detect and website\screenShot.png")
    # result = model.predict(img)
    # for r in result:
    #     boxes = r.boxes
    #     for box in boxes:
    #         b = box.xyxy[0]
    #         c = box.cls
    #         if int(c) == 0:
    #             # print(b)
    #             x, y, w, h = int(b[0]), int(b[1]), int(b[2]), int(b[3])
    #             # cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    tracker = cv.TrackerCSRT_create()
    tracker.init(frame, [x, y, w, h])
    success, bbox = tracker.update(frame)
    # Draw the bounding box around the tracked object
    if success:
        x, y, w, h = [int(i) for i in bbox]
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)