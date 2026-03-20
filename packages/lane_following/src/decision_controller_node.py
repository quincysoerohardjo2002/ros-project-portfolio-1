#!/usr/bin/env python3

import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import LanePose, Twist2DStamped


class DecisionControllerNode(DTROS):
    """Simple P-controller that converts a LanePose into wheel commands.

    Uses relative topic names — the launch file remaps them to the
    correct vehicle-specific absolute topics.

    Subscribes (relative):
        ~lane_pose      -> remapped from lane detector output
    Publishes (relative):
        ~car_cmd        -> remapped to /<veh>/car_cmd_switch_node/cmd
    """

    def __init__(self, node_name):
        super(DecisionControllerNode, self).__init__(
            node_name=node_name, node_type=NodeType.CONTROL
        )

        # --- P-controller gains (tuneable via rosparam) -------------------
        self.v_base = rospy.get_param("~v_base", 0.15)
        self.v_cautious = rospy.get_param("~v_cautious", 0.08)
        self.k_d = rospy.get_param("~k_d", -1.5)
        self.k_phi = rospy.get_param("~k_phi", -0.75)
        # Deadband: ignore d / phi below these thresholds (drive straight)
        self.d_deadband = rospy.get_param("~d_deadband", 0.05)
        self.phi_deadband = rospy.get_param("~phi_deadband", 0.03)
        # Maximum absolute steering angle (prevents wild swerving)
        self.omega_max = rospy.get_param("~omega_max", 2.0)
        # Tighter clamp when detection is uncertain
        self.omega_max_cautious = rospy.get_param("~omega_max_cautious", 0.8)
        # Smoothing factor for omega output (0 = keep old, 1 = no smoothing)
        self.omega_alpha = rospy.get_param("~omega_alpha", 0.2)

        # --- smoothed omega state -----------------------------------------
        self._omega_smooth = 0.0

        # --- subscriber (relative, remapped in launch) --------------------
        self.sub_lane_pose = rospy.Subscriber(
            "~lane_pose",
            LanePose,
            self.cb_lane_pose,
            queue_size=1,
        )

        # --- publisher (relative, remapped in launch) ---------------------
        self.pub_car_cmd = rospy.Publisher(
            "~car_cmd",
            Twist2DStamped,
            queue_size=1,
        )

        self.log("Decision controller node initialized.")

    # ------------------------------------------------------------------
    def cb_lane_pose(self, msg):
        """Receive LanePose, compute P-control, publish Twist2DStamped."""
        confident = msg.in_lane  # True if at least one line detected

        # Apply deadband: ignore tiny errors → drive straight
        d = msg.d if abs(msg.d) > self.d_deadband else 0.0
        phi = msg.phi if abs(msg.phi) > self.phi_deadband else 0.0

        # P-controller: omega = k_d * d  +  k_phi * phi
        omega_raw = self.k_d * d + self.k_phi * phi

        # Clamp omega (tighter when uncertain to prevent wild recovery)
        clamp = self.omega_max if confident else self.omega_max_cautious
        omega_clamped = max(-clamp, min(clamp, omega_raw))

        # EMA smoothing on omega
        self._omega_smooth += self.omega_alpha * (omega_clamped - self._omega_smooth)

        # Speed: slow down when detection is uncertain
        v = self.v_base if confident else self.v_cautious

        cmd = Twist2DStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.v = v
        cmd.omega = self._omega_smooth
        self.pub_car_cmd.publish(cmd)


if __name__ == "__main__":
    node = DecisionControllerNode(node_name="decision_controller_node")
    rospy.spin()
