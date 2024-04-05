#!/usr/bin/env python3

import torch
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import rospkg
from ultralytics import RTDETR
import time

class ObjectDetection(object):
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node("object_detect", anonymous=True)
        rospy.Subscriber("/usb_cam/image_raw", Image, self.update_frame_callback)
        rospy.wait_for_message("/usb_cam/image_raw", Image)
        self.model = RTDETR("/home/maxmurr/girl/runs/train/train15/weights/best.pt")
        self.CLASS_NAMES = self.model.model.names

    def update_frame_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

    def main(self):
        while not rospy.is_shutdown():
            frame = self.image
            height, width, channels = frame.shape

            # Perform object detection using RTDETR model
            results = self.model(frame)

            # Process the detection results
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            # Filter out low confidence detections
            mask = scores > 0.3
            boxes = boxes[mask]
            scores = scores[mask]
            class_ids = class_ids[mask]

            # Apply non-maximum suppression
            indexes = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.5, 0.4)

            # Draw bounding boxes and labels on the frame
            colors = np.random.uniform(0, 255, size=(len(self.CLASS_NAMES), 3))
            for i in indexes:
                x1, y1, x2, y2 = boxes[i].astype(int)
                label = self.CLASS_NAMES[class_ids[i]]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

            cv2.imshow("Image", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

if __name__ == "__main__":
    obj = ObjectDetection()
    obj.main()