import cv2
from utils.tracker import *
from configparser import ConfigParser
config = ConfigParser()

config.read('./cfg/config.ini')

MODEL_PROTO_NAME = config.get('model', 'model_folder') + config.get('model', 'model_proto_name')
MODEL_NAME = config.get('model', 'model_folder') + config.get('model', 'model_name')

SOURCE = config.get('data', 'test_folder') + config.get('data', 'test_video')
WIDTH = config.getint('data', 'width')
HEIGHT = config.getint('data', 'height')

SAVE_VIDEO = config.getboolean('test', 'save_video')
OUTPUT_FILE = config.get('data', 'test_folder') + config.get('test', 'output_file')
OUTPUT_FPS = config.getint('test', 'output_fps')


def main():
    model = cv2.dnn.readNetFromCaffe(MODEL_PROTO_NAME, MODEL_NAME)

    tracker = Tracker()

    cap = cv2.VideoCapture(SOURCE)

    if SAVE_VIDEO:
        video_writer = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'MJPG'), OUTPUT_FPS, (WIDTH, HEIGHT))

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), (104.0, 177.0, 123.0))
            model.setInput(blob)

            detections = model.forward()
            rects = []

            for i in range(0, detections.shape[2]):
                print(detections[0, 0, i, :])
                if detections[0, 0, i, 2] > 0.75:
                    box = detections[0, 0, i, 3:7] * np.array([WIDTH, HEIGHT, WIDTH, HEIGHT])
                    rects.append(box.astype("int"))

                    (startX, startY, endX, endY) = box.astype("int")

                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  (0, 255, 0), 2)

            objects = tracker.update(rects)

            for (objectID, centroid) in objects.items():
                text = "Obj {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            cv2.imshow("Frame", frame)

            if SAVE_VIDEO:
                video_writer.write(frame)

        key = cv2.waitKey(10)
        if key == ord('q'):
            break

    cap.release()
    if SAVE_VIDEO:
        video_writer.release()


if __name__ == '__main__':
    main()
