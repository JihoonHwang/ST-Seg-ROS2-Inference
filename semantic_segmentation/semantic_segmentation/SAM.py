#!/usr/bin/env python3

import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

class Segmentation(Node):
    def __init__(self):
        super().__init__("semantic_segmentation")

        self.set_config()

        self.get_logger().info("Loading Model---->")
        sam = sam_model_registry["vit_b"](checkpoint="/workspace/ROSMOS_ws/src/semantic_segmentation/library/segment-anything/ckpts/sam_vit_b_01ec64.pth")
        self.model = SamAutomaticMaskGenerator(sam)

        self.qos_profile = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        self.set_subscriber()
        self.get_logger().info("Setting Publisher---->")
        self.publisher = self.create_publisher(CompressedImage, self.seg_topic, self.qos_profile)


    def set_config(self):
        self.get_logger().info("Loading Configuration---->")

        self.declare_parameter("agent", "")
        self.declare_parameter("subscribe.rgb_topic", "/zed_front/front/left/color/compressed")
        self.declare_parameter("publish.seg_topic", "image/front/seg/compressed")
        self.declare_parameter("publish.resolution", [640, 360])

        self.agent = self.get_parameter("agent").value
        self.rgb_topic = self.agent + self.get_parameter("subscribe.rgb_topic").value
        self.seg_topic = self.agent + self.get_parameter("publish.seg_topic").value
        self.seg_resolution = self.get_parameter("publish.resolution").value

        self.get_logger().info(f"RGB Topic: {self.rgb_topic}")
        self.get_logger().info(f"Segmentation Topic: {self.seg_topic}")
        self.get_logger().info(f"Segmentation Resolution: {self.seg_resolution}")


    def set_subscriber(self):
        self.get_logger().info("Setting Subscriber---->")

        self.img_msg_type = None
        while self.img_msg_type is None:
            for topic_name, topic_type in self.get_topic_names_and_types():
                if topic_name == self.rgb_topic:
                    self.get_logger().info(f"Found topic_name: {topic_name}, topic_type: {topic_type}")
                    if topic_type[0] == "sensor_msgs/msg/Image":
                        self.img_msg_type = Image
                    elif topic_type[0] == "sensor_msgs/msg/CompressedImage":
                        self.img_msg_type = CompressedImage
                    else:
                        self.get_logger().error("Unknown ROS2 image message type")
                    break
            if self.img_msg_type is None:
                self.get_logger().info(f"Finding... topic_name: {self.rgb_topic}")
                time.sleep(1)

        self.subcription = self.create_subscription(
            self.img_msg_type,
            self.rgb_topic,
            self.callback,
            self.qos_profile
        )


    def process_img(self, msg, side):
        if self.img_msg_type == Image:
            img = CvBridge().imgmsg_to_cv2(msg, "bgr8")
        elif self.img_msg_type == CompressedImage:
            img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        else:
            self.get_logger().error("Unknown ROS2 image message type")
            return

        if img.shape[:2][::-1] != tuple(self.seg_resolution):
            img = cv2.resize(img, self.seg_resolution)

        return img


    def callback(self, msg):
        start_total = time.time()
        rgb_img = self.process_img(msg, "left")

        start_inference = time.time()
        masks = self.model.generate(rgb_img)

        opacity = 0.5
        overlay = rgb_img.copy()
        for mask in masks:
            m = mask["segmentation"].astype(bool)
            color = np.random.randint(0, 256, 3)
            overlay[m] = (1 - opacity) * overlay[m] + opacity * color
        color_seg = overlay.astype(np.uint8)
        self.get_logger().info(f"# masks: {len(masks)}")

        # result = inference_segmentor(self.model, rgb_img)
        self.get_logger().info(f"Inference Hz: {1/(time.time()-start_inference):.2f}")


        seg_msg = CompressedImage()
        seg_msg.header = msg.header
        seg_msg.format = "png"
        seg_msg.data = np.array(cv2.imencode('.png', color_seg)[1]).tobytes()
        self.publisher.publish(seg_msg)
        self.get_logger().info(f"Total Hz: {1/(time.time()-start_total):.2f}\n")


def main():
    rclpy.init()

    node = Segmentation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down segmentation node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
