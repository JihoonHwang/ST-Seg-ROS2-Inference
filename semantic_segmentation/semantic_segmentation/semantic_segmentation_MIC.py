#!/usr/bin/env python3

import time

import numpy as np
import cv2
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage

import mmcv
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint
from mmseg.apis import inference_segmentor


class SemanticSegmentation(Node):
    def __init__(self):
        super().__init__("semantic_segmentation")

        self.set_config()

        self.get_logger().info("Loading Model---->")
        self.model = self.load_model(self.model_config)

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
        self.declare_parameter("subscribe.rgb_topic", "")
        self.declare_parameter("publish.seg_topic", "")
        self.declare_parameter("publish.resolution", [0, 0])
        self.declare_parameter("model.device", "")
        self.declare_parameter("model.config", "")
        self.declare_parameter("model.weight", "")
        self.declare_parameter("model.palette", [0])

        self.agent = self.get_parameter("agent").value
        self.rgb_topic = self.agent + self.get_parameter("subscribe.rgb_topic").value
        self.seg_topic = self.agent + self.get_parameter("publish.seg_topic").value
        self.seg_resolution = self.get_parameter("publish.resolution").value
        self.device = self.get_parameter("model.device").value
        self.model_config = self.get_parameter("model.config").value
        self.model_weight = self.get_parameter("model.weight").value
        self.palette = np.array(
            self.get_parameter("model.palette").value
        ).reshape(-1, 3)

        self.get_logger().info(f"RGB Topic: {self.rgb_topic}")
        self.get_logger().info(f"Segmentation Topic: {self.seg_topic}")
        self.get_logger().info(f"Segmentation Resolution: {self.seg_resolution}")
        self.get_logger().info(f"Model Device: {self.device}")
        self.get_logger().info(f"Model Config: {self.model_config}")
        self.get_logger().info(f"Model Weight: {self.model_weight}")
        self.get_logger().info(f"Model Palette: {self.palette}")


    def load_model(self, model_config):
        model_config = mmcv.Config.fromfile(model_config)

        model_config.model.pretrained = None
        model_config.data.test_mode = True
        model_config.model.train_cfg = None

        model = build_segmentor(model_config.model, test_cfg=model_config.get('test_cfg'))
        checkpoint = load_checkpoint(
            model,
            self.model_weight,
            map_location=self.device,
            revise_keys=[(r'^module\.', ''), ('model.', '')]
        )

        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']

        model.cfg = model_config

        model = model.to(self.device)
        model.eval()

        return model


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

        # SIZE = (1280, 720)
        # K1 = np.array([[972.5252333,   0.,         656.44109101],
        #                [  0.,         970.05301359, 373.55766789],
        #                [  0.,           0.,           1.        ]])
        # D1 = np.array([-4.48675644e-01,  2.66138261e-01,
        #                -4.32486786e-05,  1.30599141e-06,
        #                -1.32859199e-01])
        # K2 = np.array([[970.11126559,   0.,         662.77072477],
        #                [  0.,         967.69449732, 379.58788643],
        #                [  0.,           0.,           1.        ]])
        # D2 = np.array([-4.54908918e-01,  3.64024995e-01,
        #                -7.01286188e-04,  3.50972545e-04,
        #                -6.02587493e-01])
        # R1 = np.array([
        #     [9.99040102e-01, -1.10250352e-03, 4.37910927e-02],
        #     [1.13434187e-03, 9.99999110e-01, -7.02208543e-04],
        #     [-4.37902795e-02, 7.51208564e-04, 9.99040463e-01]
        # ])
        # R2 = np.array([
        #     [9.98987726e-01, -2.14635438e-04, 4.49830749e-02],
        #     [1.81929670e-04, 9.99999716e-01, 7.31160979e-04],
        #     [-4.49832190e-02, -7.22237088e-04, 9.98987482e-01]
        # ])
        # P1 = np.array([
        #     [1.21764809e+03, 0.00000000e+00, 4.26123672e+02, 0.00000000e+00],
        #     [0.00000000e+00, 1.21764809e+03, 4.08569038e+02, 0.00000000e+00],
        #     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]
        # ])
        # P2 = np.array([
        #     [1.21764809e+03, 0.00000000e+00, 4.26123672e+02, -1.46433034e+05],
        #     [0.00000000e+00, 1.21764809e+03, 4.08569038e+02, 0.00000000e+00],
        #     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]
        # ])
        # map1_x, map1_y = cv2.initUndistortRectifyMap(
        #     K1, D1, R1, P1, SIZE, cv2.CV_32FC1)
        # map2_x, map2_y = cv2.initUndistortRectifyMap(
        #     K2, D2, R2, P2, SIZE, cv2.CV_32FC1)

        # if side == "left":
        #     img = cv2.remap(img, map1_x, map1_y, cv2.INTER_LINEAR)
        # else:
        #     img = cv2.remap(img, map2_x, map2_y, cv2.INTER_LINEAR)

        # Resize image
        if img.shape[:2][::-1] != tuple(self.seg_resolution):
            img = cv2.resize(img, self.seg_resolution)

        return img


    def callback(self, msg):
        start_total = time.time()
        rgb_img = self.process_img(msg, "left")

        palette_bgr = self.palette[:, ::-1]

        start_inference = time.time()
        result = inference_segmentor(self.model, rgb_img)
        self.get_logger().info(f"Inference Hz: {1/(time.time()-start_inference):.2f}")

        sem_seg = np.array(result[0])
        color_seg = palette_bgr[sem_seg]

        seg_msg = CompressedImage()
        seg_msg.header = msg.header
        seg_msg.format = "png"
        seg_msg.data = np.array(cv2.imencode('.png', color_seg)[1]).tobytes()
        self.publisher.publish(seg_msg)
        self.get_logger().info(f"Total Hz: {1/(time.time()-start_total):.2f}\n")


def main():
    rclpy.init()

    node = SemanticSegmentation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down semantic segmentation node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
