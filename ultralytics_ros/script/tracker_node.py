#!/usr/bin/env python3

import cv_bridge
import roslib.packages
import rospy
import numpy
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
import cv2
from std_msgs.msg import Int32



class TrackerNode:
    def __init__(self):
        yolo_model = rospy.get_param("~yolo_model", "yolov8n.pt")
        detection_topic = rospy.get_param("~detection_topic", "detection_result")
        image_topic = rospy.get_param("~image_topic", "image_raw")
        self.conf_thres = rospy.get_param("~conf_thres", 0.25)
        self.iou_thres = rospy.get_param("~iou_thres", 0.45)
        self.max_det = rospy.get_param("~max_det", 300)
        self.classes = rospy.get_param("~classes", None)
        self.tracker = rospy.get_param("~tracker", "botsort.yaml")
        self.debug = rospy.get_param("~debug", False)
        self.debug_conf = rospy.get_param("~debug_conf", True)
        self.debug_line_width = rospy.get_param("~debug_line_width", None)
        self.debug_font_size = rospy.get_param("~debug_font_size", None)
        self.debug_font = rospy.get_param("~debug_font", "Arial.ttf")
        self.debug_labels = rospy.get_param("~debug_labels", True)
        self.debug_boxes = rospy.get_param("~debug_boxes", True)
        path = roslib.packages.get_pkg_dir("ultralytics_ros")
        self.model = YOLO(f"{path}/models/{yolo_model}")
        self.sub = rospy.Subscriber(
            image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24
        )
        self.image_pub = rospy.Publisher("debug_image", Image, queue_size=1)
        self.detection_pub = rospy.Publisher(
            detection_topic, Detection2DArray, queue_size=1
        )
        self.bridge = cv_bridge.CvBridge()

        self.target_id_pub = rospy.Publisher('current_target_id', Int32, queue_size=1)

    def image_callback(self, msg):
        header = msg.header
        numpy_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        results = self.model.track(
            source=numpy_image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            classes=0,#self.classes,
            tracker=self.tracker,
            verbose=False,
            persist = True
        )
        self.publish_detection(results, header)
        self.publish_debug_image(results)

    # def publish_debug_image(self, results):
    #     if self.debug and results is not None:
    #         plotted_image = results[0].plot(
    #             conf=self.debug_conf,
    #             line_width=self.debug_line_width,
    #             font_size=self.debug_font_size,
    #             font=self.debug_font,
    #             labels=self.debug_labels,
    #             boxes=self.debug_boxes,
    #         )
    #         debug_image_msg = self.bridge.cv2_to_imgmsg(plotted_image, encoding="bgr8")
    #         self.image_pub.publish(debug_image_msg)

    # working-----
    # def publish_debug_image(self, results):
    #     if self.debug and results is not None:
    #         plotted_image = results[0].plot(
    #             conf=self.debug_conf,
    #             line_width=self.debug_line_width,
    #             font_size=self.debug_line_width,
    #             font=self.debug_font,
    #             labels=self.debug_labels,
    #             boxes=self.debug_boxes,
    #         )
            
    #         # Convert to numpy array for OpenCV manipulation
    #         plotted_image = numpy.array(plotted_image)
            
    #         # Check if we have a target ID to highlight
    #         try:
    #             target_id = rospy.wait_for_message('target_id_to_highlight', Int32, timeout=0.1)
    #             if results[0].boxes.id is not None:
    #                 for i, box_id in enumerate(results[0].boxes.id):
    #                     if box_id.item() == target_id.data:
    #                         bbox = results[0].boxes.xyxy[i].cpu().numpy().astype(int)
    #                         x1, y1, x2, y2 = bbox
                            
    #                         # Draw thicker red rectangle
    #                         cv2.rectangle(plotted_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            
    #                         # Add "Target" text
    #                         text = f"Target (ID: {target_id.data})"
    #                         text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    #                         text_x = int((x1 + x2) / 2 - text_size[0] / 2)
    #                         text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                            
    #                         cv2.putText(plotted_image, text, (text_x, text_y),
    #                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #                         break
    #         except rospy.ROSException:
    #             pass  # No target ID message received
            
    #         debug_image_msg = self.bridge.cv2_to_imgmsg(plotted_image, encoding="bgr8")
    #         self.image_pub.publish(debug_image_msg)

    def publish_debug_image(self, results):
        if self.debug and results is not None:
            try:
                # Get the original image
                plotted_image = results[0].orig_img.copy()
                
                # Get target ID if available
                target_id = None
                try:
                    target_id_msg = rospy.wait_for_message('target_id_to_highlight', Int32, timeout=0.1)
                    target_id = target_id_msg.data
                    # print(f"Highlighting target ID: {target_id}")
                except rospy.ROSException:
                    # print("No target ID received - highlighting first detection")
                    if results[0].boxes.id is not None and len(results[0].boxes.id) > 0:
                        target_id = results[0].boxes.id[0].item()

                # Draw all detections
                for i, box in enumerate(results[0].boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf.item()
                    cls_id = box.cls.item()
                    box_id = box.id.item() if box.id is not None else -1
                    
                    # Default green box for all detections
                    color = (0, 255, 0)  # Green
                    thickness = 2
                    
                    # Highlight the target detection
                    if box_id == target_id:
                        color = (0, 0, 255)  # Red
                        thickness = 3
                        
                        # Add "Target" label
                        label = f"Target (ID: {box_id}, {conf:.2f})"
                        (label_width, label_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(plotted_image, 
                                    (x1, y1 - label_height - baseline - 5), 
                                    (x1 + label_width, y1), 
                                    color, -1)  # Filled background
                        cv2.putText(plotted_image, label, 
                                    (x1, y1 - baseline - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                    (255, 255, 255), 2)  # White text
                    
                    # Draw the bounding box
                    cv2.rectangle(plotted_image, (x1, y1), (x2, y2), color, thickness)
                    
                    # Add class label for all detections
                    class_label = f"{self.model.names[int(cls_id)]} {conf:.2f}"
                    cv2.putText(plotted_image, class_label, 
                                (x1, y2 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                color, 2)
                
                debug_image_msg = self.bridge.cv2_to_imgmsg(plotted_image, encoding="bgr8")
                self.image_pub.publish(debug_image_msg)
                
            except Exception as e:
                print(f"Error in debug image processing: {str(e)}")

    def publish_detection(self, results, header):
        if results is not None:
            detections_msg = Detection2DArray()
            detections_msg.header = header
            bounding_box = results[0].boxes.xywh
            classes = results[0].boxes.cls
            confidence_score = results[0].boxes.conf
            identity = results[0].boxes.id
            if identity is None:
            	identity = [0,0,0,0,0,0,0]
            #detection_id = 0
            #if results[0].boxes.id is not None:
            #    detection_id = results[0].boxes.id.cpu().numpy().astype(int)       
            #print("Identity", identity)
            for bbox, cls, conf, ids in zip(bounding_box, classes, confidence_score, identity):
                detection = Detection2D()
                #print("DETECTION_ID", ids)
                detection.bbox.center.x = float(bbox[0])
                detection.bbox.center.y = float(bbox[1])
                detection.bbox.size_x = float(bbox[2])
                detection.bbox.size_y = float(bbox[3])
                #detection.id = int(ids)
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.id = int(ids) #int(cls) #
                hypothesis.score = float(conf)
                detection.results.append(hypothesis)
                detections_msg.detections.append(detection)
            self.detection_pub.publish(detections_msg)


if __name__ == "__main__":
    rospy.init_node("tracker_node")
    node = TrackerNode()
    rospy.spin()
