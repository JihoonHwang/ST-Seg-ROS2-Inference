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

# Updated imports for MMSegmentation 1.2.2
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmseg.apis import init_model, inference_model


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
        # print(self.palette)  
        self.get_logger().info(f"RGB Topic: {self.rgb_topic}")
        self.get_logger().info(f"Segmentation Topic: {self.seg_topic}")
        self.get_logger().info(f"Segmentation Resolution: {self.seg_resolution}")
        self.get_logger().info(f"Model Device: {self.device}")
        self.get_logger().info(f"Model Config: {self.model_config}")
        self.get_logger().info(f"Model Weight: {self.model_weight}")
        self.get_logger().info(f"Model Palette: {self.palette}")


    def load_model(self, model_config):
        # Use init_model API for MMSegmentation 1.2.2
        model = init_model(model_config, self.model_weight, device=self.device)
        
        # Load checkpoint to get metadata
        checkpoint = load_checkpoint(
            model,
            self.model_weight,
            map_location=self.device,
            revise_keys=[(r'^module\.', ''), ('model.', '')]
        )
        
        # Set CLASSES and PALETTE from checkpoint metadata
        if 'meta' in checkpoint:
            if 'CLASSES' in checkpoint['meta']:
                model.CLASSES = checkpoint['meta']['CLASSES']
            if 'PALETTE' in checkpoint['meta']:
                model.PALETTE = checkpoint['meta']['PALETTE']
        
        # For compatibility, store the config
        model.cfg = Config.fromfile(model_config)
        
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

        # # Resize image
        # if img.shape[:2][::-1] != tuple(self.seg_resolution):
        #     img = cv2.resize(img, self.seg_resolution)

        return img


    def callback(self, msg):
        start_total = time.time()
        rgb_img = self.process_img(msg, "left")

        palette_bgr = self.palette[:, ::-1]

        start_inference = time.time()
        # Use inference_model API for MMSegmentation 1.2.2
        result = inference_model(self.model, rgb_img)
        self.get_logger().info(f"Inference Hz: {1/(time.time()-start_inference):.2f}")

        # Extract semantic segmentation result
        # In MMSegmentation 1.2.2, result is a SegDataSample object
        if hasattr(result, 'pred_sem_seg'):
            sem_seg = result.pred_sem_seg.data.cpu().numpy()[0]
        else:
            # Fallback for compatibility
            sem_seg = result[0] if isinstance(result, (list, tuple)) else result
            if isinstance(sem_seg, torch.Tensor):
                sem_seg = sem_seg.cpu().numpy()
            if len(sem_seg.shape) == 3 and sem_seg.shape[0] == 1:
                sem_seg = sem_seg[0]

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