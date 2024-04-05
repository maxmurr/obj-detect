import torch
import numpy as np
import cv2
import time
from ultralytics import RTDETR
import supervision as sv

class DETRClass:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print('Using device:', self.device)

        self.model = RTDETR("/home/maxmurr/girl/girl-obj-detection/weights/best.pt")
        self.CLASS_NAMES_DICT = self.model.names
        print("Classes:", self.CLASS_NAMES_DICT)

        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=3, text_thickness=3, text_scale=1.5)

    def plot_bboxes(self, results, frame):
        boxes = results[0].boxes
        class_id = boxes.cls.cpu().numpy().astype(np.int32)
        conf = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        detections = sv.Detections(xyxy=xyxy, class_id=class_id, confidence=conf)

        labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:.2f}" for _, _, confidence, class_id, _ in detections]
        frame = self.box_annotator(frame, detections, labels)

        return frame

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened(), f"Failed to open camera {self.capture_index}"

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while cap.isOpened():
            start_time = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame)
            frame = self.plot_bboxes(results, frame)

            end_time = time.perf_counter()
            fps = 1 / (end_time - start_time)

            cv2.putText(frame, f"FPS: {fps:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("DETR", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    transform_detector = DETRClass(0)
    transform_detector()