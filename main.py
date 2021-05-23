import cv2
from utils.tracker import *


def main():
    model_folder = "./model/"
    model_proto_name = model_folder + "weights-prototxt.txt"
    model_name = model_folder + "res_ssd_300Dim.caffeModel"

    model = cv2.dnn.readNetFromCaffe(model_proto_name, model_name)

    tracker = Tracker()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            model.setInput(blob)

            detections = model.forward()
            rects = []

            for i in range(0, detections.shape[2]):
                
                if detections[0, 0, i, 2] > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([1280, 720, 1280, 720])
                    rects.append(box.astype("int"))

                    (startX, startY, endX, endY) = box.astype("int")

                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  (0, 255, 0), 2)

            objects = tracker.update(rects)

            for (objectID, centroid) in objects.items():
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            cv2.imshow("Frame", frame)

        key = cv2.waitKey(10)
        if key == ord('q'):
            break


if __name__ == '__main__':
    main()
