#!/usr/bin/env python3
"""
Line Detector Node — Perception
================================
Based on the Duckietown lane-following demo's line_detector_node and
lane_filter_node.  Detects white and yellow lane markings using HSV
filtering combined with Canny edge detection and Hough line segments,
then estimates the lane pose (d, phi).

DEMO BASIS (from dt-core line_detector + lane_filter):
  - HSV colour segmentation for white and yellow lane lines
  - Canny edge detection on colour-filtered masks
  - Probabilistic Hough transform to extract line segments
  - Segment-based lane pose estimation (d = lateral offset, phi = heading)

FINE-TUNING (our additions for smoother driving):
  - Morphological cleanup to reduce noise in colour masks
  - Weighted segment averaging using segment length
  - EMA (exponential moving average) smoothing on d and phi
  - Confidence levels (2 = both lines, 1 = single line, 0 = none)
  - Hold-last-good fallback to coast through brief detection gaps
  - Sanity gates: minimum segment count, position checks, jump rejection

Subscribes:
    ~image/compressed              <-- camera feed (remapped in launch)

Publishes:
    ~lane_pose                     --> LanePose for the controller
    ~debug/segments/compressed     --> annotated image with detected segments
"""

import rospy
import numpy as np
import cv2
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import LanePose


class LineDetectorNode(DTROS):
    """Detects lane lines and publishes a LanePose estimate.

    Architecture follows the Duckietown demo: HSV filtering -> Canny ->
    Hough segments -> lane pose estimation.  Fine-tuned with EMA smoothing,
    confidence tracking, and noise rejection for smoother driving.
    """

    def __init__(self, node_name):
        super(LineDetectorNode, self).__init__(
            node_name=node_name, node_type=NodeType.PERCEPTION
        )

        # =================================================================
        # PARAMETERS — HSV colour ranges
        # (demo basis) The Duckietown demo uses HSV thresholds to separate
        # white (right lane edge) and yellow (centre line) markings.
        # Fine-tuned: ranges adjusted for Duckietown simulator lighting.
        # =================================================================
        self.white_lower = np.array(
            rospy.get_param("~white_lower", [0, 0, 150])
        )
        self.white_upper = np.array(
            rospy.get_param("~white_upper", [180, 60, 255])
        )
        self.yellow_lower = np.array(
            rospy.get_param("~yellow_lower", [18, 60, 100])
        )
        self.yellow_upper = np.array(
            rospy.get_param("~yellow_upper", [40, 255, 255])
        )

        # =================================================================
        # PARAMETERS — Image processing
        # (demo basis) crop_fraction, Canny thresholds, and Hough transform
        # settings follow the demo's line_detector_node approach.
        # =================================================================
        self.crop_fraction = rospy.get_param("~crop_fraction", 0.4)
        # Canny edge detection thresholds (from demo)
        self.canny_low = rospy.get_param("~canny_low", 50)
        self.canny_high = rospy.get_param("~canny_high", 150)
        # Hough transform parameters (from demo)
        self.hough_threshold = rospy.get_param("~hough_threshold", 20)
        self.hough_min_length = rospy.get_param("~hough_min_length", 10)
        self.hough_max_gap = rospy.get_param("~hough_max_gap", 5)

        # =================================================================
        # PARAMETERS — Fine-tuning additions
        # These go beyond the base demo to improve driving stability.
        # =================================================================
        # Morphological kernel size for cleaning colour masks
        self.morph_size = rospy.get_param("~morph_size", 5)
        # Minimum Hough segments required to trust a colour detection
        self.min_segments = rospy.get_param("~min_segments", 2)
        # EMA smoothing factors (lower = smoother, higher = more responsive)
        self.ema_alpha_high = rospy.get_param("~ema_alpha_high", 0.35)
        self.ema_alpha_low = rospy.get_param("~ema_alpha_low", 0.10)
        # Maximum allowed jump in d between frames (rejects outliers)
        self.max_d_jump = rospy.get_param("~max_d_jump", 0.15)
        # How many frames to coast on last-good estimate when detection drops
        self.hold_max = rospy.get_param("~hold_max", 15)
        # Expected lane width as fraction of crop width (single-line fallback)
        self.expected_lane_frac = rospy.get_param(
            "~expected_lane_frac", 0.45
        )

        # =================================================================
        # Internal state for smoothing and confidence tracking
        # =================================================================
        self._d_smooth = 0.0
        self._phi_smooth = 0.0
        # Detection confidence: 2 = both lines, 1 = one line, 0 = none
        self._confidence = 0
        # Last accepted raw d (for jump rejection)
        self._last_d_raw = None
        # Hold counter: frames remaining to coast on last-good d/phi
        self._hold_count = 0

        # =================================================================
        # ROS subscriber and publishers
        # =================================================================
        self.sub_image = rospy.Subscriber(
            "~image/compressed",
            CompressedImage,
            self.cb_image,
            queue_size=1,
            buff_size="20MB",
        )

        self.pub_lane_pose = rospy.Publisher(
            "~lane_pose", LanePose, queue_size=1
        )
        self.pub_debug_segments = rospy.Publisher(
            "~debug/segments/compressed", CompressedImage, queue_size=1
        )

        self.log("Line detector node initialized (demo-based + fine-tuned).")

    # =====================================================================
    # MAIN CALLBACK — camera image processing pipeline
    # =====================================================================
    def cb_image(self, msg):
        """Process one camera frame through the full detection pipeline.

        Pipeline (demo-inspired):
          1. Crop to road region
          2. HSV colour filtering for white and yellow
          3. Canny edge detection (from demo)
          4. Hough line segment extraction (from demo)
          5. Segment-based lane pose estimation
          6. Smoothing and confidence logic (fine-tuned)
        """
        # --- Decode compressed image ---
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            return

        h, w, _ = image.shape

        # --- Step 1: Crop to bottom portion (road area) ---
        # (demo convention) Only process the lower part of the image where
        # the road surface is visible; reduces noise from the horizon.
        crop_y = int(h * (1.0 - self.crop_fraction))
        crop = image[crop_y:, :]
        ch, cw, _ = crop.shape
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # --- Step 2: HSV colour filtering (from demo) ---
        # The demo isolates white and yellow pixels using HSV thresholds.
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)

        # (fine-tuning) Morphological open+close to remove small noise blobs
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_size, self.morph_size)
        )
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

        # --- Step 3: Canny edge detection (from demo) ---
        # The demo applies Canny on the colour-filtered masks to find edges
        # before extracting line segments.
        white_edges = cv2.Canny(white_mask, self.canny_low, self.canny_high)
        yellow_edges = cv2.Canny(
            yellow_mask, self.canny_low, self.canny_high
        )

        # --- Step 4: Probabilistic Hough transform (from demo) ---
        # The demo uses HoughLinesP to extract line segments from edges.
        # Each segment is represented as (x1, y1, x2, y2).
        white_lines = cv2.HoughLinesP(
            white_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_length,
            maxLineGap=self.hough_max_gap,
        )
        yellow_lines = cv2.HoughLinesP(
            yellow_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_length,
            maxLineGap=self.hough_max_gap,
        )

        # Reshape from HoughLinesP format (N,1,4) to (N,4)
        white_segs = (
            white_lines.reshape(-1, 4)
            if white_lines is not None
            else np.empty((0, 4))
        )
        yellow_segs = (
            yellow_lines.reshape(-1, 4)
            if yellow_lines is not None
            else np.empty((0, 4))
        )

        # --- Step 5: Segment-based lane pose estimation ---
        # (demo-inspired) Simplified version of the demo's lane_filter_node.
        # Instead of a full histogram filter with ground projection, we use
        # weighted segment midpoints and angles directly in image coords.
        d_raw, phi_raw, confidence = self._estimate_lane_pose(
            white_segs, yellow_segs, cw, ch
        )

        # --- Step 6: Smoothing and confidence (fine-tuned) ---
        self._apply_smoothing(d_raw, phi_raw, confidence)

        # --- Publish LanePose ---
        holding = self._confidence == 0 and self._hold_count > 0
        pose = LanePose()
        pose.header.stamp = rospy.Time.now()
        pose.d = self._d_smooth
        pose.phi = self._phi_smooth
        pose.in_lane = self._confidence >= 1 or holding
        self.pub_lane_pose.publish(pose)

        # --- Debug image (only if someone is listening) ---
        if self.pub_debug_segments.get_num_connections() > 0:
            self._publish_debug(crop, white_segs, yellow_segs, cw, holding)

    # =====================================================================
    # LANE POSE ESTIMATION
    # (demo-inspired) Simplified version of the demo's lane_filter_node.
    # =====================================================================
    def _estimate_lane_pose(self, white_segs, yellow_segs, cw, ch):
        """Estimate d and phi from detected line segments.

        Demo basis: the Duckietown lane_filter_node uses ground-projected
        segments in a histogram vote to find the best (d, phi).  We simplify
        this by computing weighted averages of segment positions and angles
        directly in image coordinates.

        Fine-tuning: position-based filtering, length-weighted averaging,
        and single-line fallback using expected lane width.

        Returns
        -------
        d_raw : float or None
            Lateral offset estimate (positive = shifted left of centre).
        phi_raw : float or None
            Heading error estimate (positive = heading left of lane).
        confidence : int
            0 = no lines, 1 = one line, 2 = both lines.
        """
        mid = cw / 2.0
        expected_half = cw * self.expected_lane_frac / 2.0

        # --- Filter segments by expected position ---
        # (fine-tuning) White line should be on the RIGHT side, yellow LEFT.
        white_f = self._filter_segments(white_segs, cw, side="right")
        yellow_f = self._filter_segments(yellow_segs, cw, side="left")

        # --- Compute weighted midpoint-x and angle per colour ---
        white_x, white_angle, white_ok = self._segment_stats(white_f)
        yellow_x, yellow_angle, yellow_ok = self._segment_stats(yellow_f)

        # (fine-tuning) Sanity: white must be to the right of yellow
        if white_ok and yellow_ok and white_x < yellow_x:
            white_ok = False

        # --- Estimate lane centre (demo-inspired) ---
        lane_center = None
        if white_ok and yellow_ok:
            lane_center = (white_x + yellow_x) / 2.0
            confidence = 2
        elif yellow_ok:
            lane_center = yellow_x + expected_half
            confidence = 1
        elif white_ok:
            lane_center = white_x - expected_half
            confidence = 1
        else:
            confidence = 0

        # --- Compute d (lateral offset, normalised to [-1, 1]) ---
        if lane_center is not None:
            d_raw = (lane_center - mid) / mid
        else:
            d_raw = None

        # --- Compute phi (heading error from segment angles) ---
        # (demo-inspired) Use the angle of detected line segments to
        # estimate how much the robot's heading deviates from the lane.
        if confidence == 2:
            phi_raw = -(white_angle + yellow_angle) / 2.0
        elif confidence == 1:
            angle = yellow_angle if yellow_ok else white_angle
            phi_raw = -angle
        else:
            phi_raw = None

        return d_raw, phi_raw, confidence

    def _filter_segments(self, segments, cw, side):
        """Filter segments by expected lateral position.

        (fine-tuning) Reject segments that appear on the wrong side of the
        image — for example a yellow-like reflection on the right.
        """
        if len(segments) == 0:
            return segments

        # Midpoint x of each segment
        mid_x = (segments[:, 0] + segments[:, 2]) / 2.0

        if side == "right":
            mask = mid_x > cw * 0.30
        else:  # left
            mask = mid_x < cw * 0.70

        return segments[mask]

    def _segment_stats(self, segments):
        """Compute length-weighted average x-position and angle.

        (demo basis) The demo weighs segment contributions; we use segment
        length as weight for a robust average.

        Returns
        -------
        avg_x : float
            Weighted average midpoint x-coordinate.
        avg_angle : float
            Weighted average angle from vertical (radians).
        valid : bool
            True if enough segments are present.
        """
        if len(segments) < self.min_segments:
            return 0.0, 0.0, False

        x1 = segments[:, 0].astype(float)
        y1 = segments[:, 1].astype(float)
        x2 = segments[:, 2].astype(float)
        y2 = segments[:, 3].astype(float)

        # Segment lengths (longer segments are more reliable)
        lengths = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total_length = lengths.sum()
        if total_length < 1e-6:
            return 0.0, 0.0, False

        weights = lengths / total_length

        # Weighted average midpoint x
        mid_x = (x1 + x2) / 2.0
        avg_x = float(np.sum(mid_x * weights))

        # Weighted average angle from vertical
        # atan2(dx, dy) gives the angle from the vertical axis;
        # we normalise direction so segments always "point upward"
        # in image coordinates (decreasing y).
        dx = x2 - x1
        dy = y2 - y1
        flip = dy > 0
        dx = np.where(flip, -dx, dx)
        dy = np.where(flip, -dy, dy)
        angles = np.arctan2(dx, -dy)
        avg_angle = float(np.sum(angles * weights))

        return avg_x, avg_angle, True

    # =====================================================================
    # SMOOTHING — fine-tuned additions for stable driving
    # =====================================================================
    def _apply_smoothing(self, d_raw, phi_raw, confidence):
        """Apply EMA smoothing, hold-last-good, and jump rejection.

        (fine-tuning) These additions are not in the base demo but are
        essential for smooth driving on the Duckiebot:
          - EMA smoothing prevents abrupt lane-pose changes
          - Hold-last-good allows coasting through brief detection gaps
          - Jump rejection filters spurious measurements (e.g. intersections)
        """
        self._confidence = confidence

        # --- Jump rejection ---
        if d_raw is not None and self._last_d_raw is not None:
            if abs(d_raw - self._last_d_raw) > self.max_d_jump:
                self._confidence = 0
                d_raw = None
                phi_raw = None

        # --- Update hold counter ---
        holding = False
        if self._confidence >= 1 and d_raw is not None:
            self._last_d_raw = d_raw
            self._hold_count = self.hold_max
        elif self._hold_count > 0:
            self._hold_count -= 1
            holding = True

        # --- Determine raw values for smoothing ---
        if d_raw is not None:
            d_val = d_raw
            phi_val = phi_raw if phi_raw is not None else 0.0
        elif holding:
            # Coast on last smooth values during hold period
            d_val = self._d_smooth
            phi_val = self._phi_smooth
        else:
            # No detection and hold exhausted: decay toward zero
            d_val = self._d_smooth * 0.8
            phi_val = self._phi_smooth * 0.8

        # --- EMA smoothing ---
        # Faster when confident (alpha_high), slower when unsure (alpha_low)
        if holding:
            alpha = 0.0  # freeze smoothed values during hold
        elif self._confidence >= 2:
            alpha = self.ema_alpha_high
        else:
            alpha = self.ema_alpha_low

        self._d_smooth += alpha * (d_val - self._d_smooth)
        self._phi_smooth += alpha * (phi_val - self._phi_smooth)

    # =====================================================================
    # DEBUG VISUALISATION
    # =====================================================================
    def _publish_debug(self, crop, white_segs, yellow_segs, cw, holding):
        """Publish an annotated debug image showing detected segments."""
        overlay = crop.copy()
        ch = overlay.shape[0]

        # Draw yellow segments (bright yellow)
        for seg in yellow_segs:
            cv2.line(
                overlay,
                (int(seg[0]), int(seg[1])),
                (int(seg[2]), int(seg[3])),
                (0, 255, 255),
                2,
            )

        # Draw white segments (bright white)
        for seg in white_segs:
            cv2.line(
                overlay,
                (int(seg[0]), int(seg[1])),
                (int(seg[2]), int(seg[3])),
                (255, 255, 255),
                2,
            )

        # Draw image centre reference line (grey)
        mid = cw // 2
        cv2.line(overlay, (mid, 0), (mid, ch), (128, 128, 128), 1)

        # Overlay smoothed d, phi, confidence, and hold status
        conf_label = ["NONE", "LOW", "HIGH"][self._confidence]
        hold_str = "  HOLD" if holding else ""
        info = (
            f"d={self._d_smooth:.2f}  phi={self._phi_smooth:.2f}"
            f"  [{conf_label}]{hold_str}"
        )
        cv2.putText(
            overlay,
            info,
            (5, ch - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )

        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode(".jpg", overlay)[1]).tobytes()
        self.pub_debug_segments.publish(msg)


if __name__ == "__main__":
    node = LineDetectorNode(node_name="line_detector_node")
    rospy.spin()
