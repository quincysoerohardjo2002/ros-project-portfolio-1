#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Float32


class DuckieDetectionNode(DTROS):
    """Node for detecting duckies (rubber ducks) on the road.

    Subscribes to the camera image, detects yellow rubber ducks,
    and publishes detection status and estimated distance.
    """

    def __init__(self, node_name):
        super(DuckieDetectionNode, self).__init__(
            node_name=node_name, node_type=NodeType.PERCEPTION
        )

        # Get vehicle name from environment
        self.veh = rospy.get_param("~veh", "joepduckiebot")

        # Subscriber: compressed camera image
        self.sub_image = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.cb_image,
            queue_size=1,
            buff_size="20MB",
        )

        # Publisher: duckie detected (boolean)
        self.pub_duckie_detected = rospy.Publisher(
            f"/{self.veh}/duckie_detection_node/detected",
            Bool,
            queue_size=1,
        )

        # Publisher: estimated distance to duckie
        self.pub_duckie_distance = rospy.Publisher(
            f"/{self.veh}/duckie_detection_node/distance",
            Float32,
            queue_size=1,
        )

        self.log("Duckie detection node initialized.")

    def cb_image(self, msg):
        """Callback for camera images. Placeholder for duckie detection."""
        # TODO: Implement duckie detection pipeline
        # 1. Decode compressed image
        # 2. Convert to HSV color space
        # 3. Filter for yellow (duckie color) regions
        # 4. Apply morphological operations to clean mask
        # 5. Find contours, filter by area
        # 6. Estimate distance based on bounding box size
        # 7. Publish detection status and distance
        pass

    def publish_detection(self, detected, distance=0.0):
        """Publish duckie detection status and distance."""
        det_msg = Bool()
        det_msg.data = detected
        self.pub_duckie_detected.publish(det_msg)

        dist_msg = Float32()
        dist_msg.data = distance
        self.pub_duckie_distance.publish(dist_msg)


if __name__ == "__main__":
    node = DuckieDetectionNode(node_name="duckie_detection_node")
    rospy.spin()
