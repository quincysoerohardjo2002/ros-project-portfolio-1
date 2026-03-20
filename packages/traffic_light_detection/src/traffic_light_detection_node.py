#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String


class TrafficLightDetectionNode(DTROS):
    """Node for detecting traffic lights in Duckietown.

    Subscribes to the camera image, detects traffic light colors
    (red, green), and publishes the detected state.
    """

    def __init__(self, node_name):
        super(TrafficLightDetectionNode, self).__init__(
            node_name=node_name, node_type=NodeType.PERCEPTION
        )

        # Get vehicle name from environment
        self.veh = rospy.get_param("~veh", "joepduckiebot")

        # Current detected traffic light state
        self.light_state = "unknown"

        # Subscriber: compressed camera image
        self.sub_image = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.cb_image,
            queue_size=1,
            buff_size="20MB",
        )

        # Publisher: detected traffic light state
        self.pub_light_state = rospy.Publisher(
            f"/{self.veh}/traffic_light_detection_node/light_state",
            String,
            queue_size=1,
        )

        self.log("Traffic light detection node initialized.")

    def cb_image(self, msg):
        """Callback for camera images. Placeholder for traffic light detection."""
        # TODO: Implement traffic light detection pipeline
        # 1. Decode compressed image
        # 2. Convert to HSV color space
        # 3. Filter for red and green regions
        # 4. Apply contour detection / bounding box
        # 5. Classify traffic light state
        # 6. Publish state
        pass

    def publish_state(self, state):
        """Publish the detected traffic light state."""
        msg = String()
        msg.data = state
        self.pub_light_state.publish(msg)


if __name__ == "__main__":
    node = TrafficLightDetectionNode(node_name="traffic_light_detection_node")
    rospy.spin()
