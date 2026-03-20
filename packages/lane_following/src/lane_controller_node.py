#!/usr/bin/env python3
"""
Lane Controller Node — Control
================================
Based on the Duckietown lane-following demo's lane_controller_node.
Receives a LanePose estimate and computes velocity and steering commands.

DEMO BASIS (from dt-core lane_controller):
  - P-control on lateral offset (d) and heading error (phi):
        omega = k_d * d  +  k_phi * phi
  - Integral terms for steady-state error correction:
        omega += k_Id * integral(d)  +  k_Iphi * integral(phi)

FINE-TUNING (our additions for smoother driving):
  - Deadband: ignore very small d/phi values to prevent jitter
  - Omega smoothing (EMA) to prevent abrupt steering changes
  - Omega clamping with tighter limits when detection is uncertain
  - Speed adaptation: slow down during uncertain detection
  - Anti-windup on integral terms to prevent overshoot

Subscribes:
    ~lane_pose  <-- LanePose from line_detector_node (remapped in launch)

Publishes:
    ~car_cmd    --> Twist2DStamped (remapped to car_cmd_switch_node)

Design note:
    This node only handles lane-following control.  A future decision node
    (for traffic lights / duckie detection) can override or gate the car_cmd
    topic without modifying this controller.
"""

import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import LanePose, Twist2DStamped


class LaneControllerNode(DTROS):
    """PI lane controller that converts LanePose into wheel commands.

    Architecture follows the Duckietown demo's lane_controller_node,
    extended with fine-tuned smoothing and safety features.
    """

    def __init__(self, node_name):
        super(LaneControllerNode, self).__init__(
            node_name=node_name, node_type=NodeType.CONTROL
        )

        # =================================================================
        # PARAMETERS — Speed settings
        # =================================================================
        # Base forward speed when detection is confident
        self.v_base = rospy.get_param("~v_base", 0.15)
        # Reduced speed when detection confidence is low
        self.v_cautious = rospy.get_param("~v_cautious", 0.08)

        # =================================================================
        # PARAMETERS — PI gains (demo basis)
        # The Duckietown demo's lane_controller uses proportional gains
        # k_d and k_phi, plus integral gains k_Id and k_Iphi.
        # =================================================================
        # Proportional gain on lateral offset d (from demo)
        self.k_d = rospy.get_param("~k_d", -1.5)
        # Proportional gain on heading error phi (from demo)
        self.k_phi = rospy.get_param("~k_phi", -0.75)
        # Integral gain on d (from demo, helps eliminate persistent offset)
        self.k_Id = rospy.get_param("~k_Id", -0.2)
        # Integral gain on phi (from demo, corrects steady-state heading)
        self.k_Iphi = rospy.get_param("~k_Iphi", -0.05)

        # =================================================================
        # PARAMETERS — Fine-tuning additions
        # =================================================================
        # Deadband: ignore errors below these thresholds (prevents jitter)
        self.d_deadband = rospy.get_param("~d_deadband", 0.05)
        self.phi_deadband = rospy.get_param("~phi_deadband", 0.03)
        # Maximum absolute steering angle (prevents wild swerving)
        self.omega_max = rospy.get_param("~omega_max", 2.0)
        # Tighter steering clamp when detection is uncertain
        self.omega_max_cautious = rospy.get_param(
            "~omega_max_cautious", 0.8
        )
        # EMA smoothing factor for omega output (0 = frozen, 1 = instant)
        self.omega_alpha = rospy.get_param("~omega_alpha", 0.2)
        # Anti-windup: maximum integral accumulator magnitude
        self.integral_max = rospy.get_param("~integral_max", 0.5)

        # =================================================================
        # Internal state
        # =================================================================
        self._omega_smooth = 0.0
        self._d_integral = 0.0
        self._phi_integral = 0.0
        self._last_time = None

        # =================================================================
        # ROS subscriber and publisher
        # =================================================================
        self.sub_lane_pose = rospy.Subscriber(
            "~lane_pose",
            LanePose,
            self.cb_lane_pose,
            queue_size=1,
        )

        self.pub_car_cmd = rospy.Publisher(
            "~car_cmd",
            Twist2DStamped,
            queue_size=1,
        )

        self.log(
            "Lane controller node initialized (demo-based PI + fine-tuned)."
        )

    # =================================================================
    # CONTROL CALLBACK
    # =================================================================
    def cb_lane_pose(self, msg):
        """Receive LanePose, compute PI control, publish Twist2DStamped.

        Demo basis:
            omega = k_d * d  +  k_phi * phi          (proportional)
                  + k_Id * int(d)  +  k_Iphi * int(phi)  (integral)

        Fine-tuning:
            deadband, EMA smoothing, omega clamping, speed adaptation
        """
        confident = msg.in_lane  # True when at least one line is detected

        # --- Compute dt for integral terms ---
        now = rospy.Time.now()
        dt = 0.0
        if self._last_time is not None:
            dt = (now - self._last_time).to_sec()
            dt = min(dt, 0.1)  # cap to prevent jumps after long pauses
        self._last_time = now

        # --- Apply deadband (fine-tuning: prevents jitter) ---
        d = msg.d if abs(msg.d) > self.d_deadband else 0.0
        phi = msg.phi if abs(msg.phi) > self.phi_deadband else 0.0

        # --- Update integral terms (demo basis + anti-windup) ---
        if confident and dt > 0:
            self._d_integral += d * dt
            self._phi_integral += phi * dt
            # (fine-tuning) Anti-windup: clamp integrals
            self._d_integral = max(
                -self.integral_max,
                min(self.integral_max, self._d_integral),
            )
            self._phi_integral = max(
                -self.integral_max,
                min(self.integral_max, self._phi_integral),
            )
        elif not confident:
            # (fine-tuning) Slowly decay integrals when not confident
            # to prevent stale buildup from affecting recovery
            self._d_integral *= 0.9
            self._phi_integral *= 0.9

        # --- PI control law (demo basis) ---
        # Proportional terms (from demo)
        omega_raw = self.k_d * d + self.k_phi * phi
        # Integral terms (from demo)
        omega_raw += (
            self.k_Id * self._d_integral
            + self.k_Iphi * self._phi_integral
        )

        # --- Clamp omega (fine-tuning: tighter when uncertain) ---
        clamp = self.omega_max if confident else self.omega_max_cautious
        omega_clamped = max(-clamp, min(clamp, omega_raw))

        # --- EMA smoothing (fine-tuning: prevents abrupt steering) ---
        self._omega_smooth += self.omega_alpha * (
            omega_clamped - self._omega_smooth
        )

        # --- Speed adaptation (fine-tuning) ---
        v = self.v_base if confident else self.v_cautious

        # --- Publish command ---
        cmd = Twist2DStamped()
        cmd.header.stamp = now
        cmd.v = v
        cmd.omega = self._omega_smooth
        self.pub_car_cmd.publish(cmd)


if __name__ == "__main__":
    node = LaneControllerNode(node_name="lane_controller_node")
    rospy.spin()
