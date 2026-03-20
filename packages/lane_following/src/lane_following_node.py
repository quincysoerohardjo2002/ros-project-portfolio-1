#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import LanePose


class LaneDetectorNode(DTROS):
    """Detects lane markings and publishes a LanePose estimate.

    Uses relative topic names so the actual camera and output topics
    are wired through <remap> in the launch file.

    Subscribes (relative):
        ~image/compressed       -> remapped to camera topic
    Publishes (relative):
        ~lane_pose              -> consumed by the decision controller
        ~debug/roi/compressed   -> cropped road region
        ~debug/mask/compressed  -> combined white+yellow mask
        ~debug/overlay/compressed -> annotated overlay with centroids
    """

    def __init__(self, node_name):
        super(LaneDetectorNode, self).__init__(
            node_name=node_name, node_type=NodeType.PERCEPTION
        )

        # --- tuneable parameters ------------------------------------------
        # HSV range for the *white* right-side lane marking
        self.white_lower = np.array(rospy.get_param("~white_lower", [0, 0, 150]))
        self.white_upper = np.array(rospy.get_param("~white_upper", [180, 50, 255]))
        # HSV range for the *yellow* centre line
        self.yellow_lower = np.array(rospy.get_param("~yellow_lower", [18, 60, 100]))
        self.yellow_upper = np.array(rospy.get_param("~yellow_upper", [40, 255, 255]))
        # Crop: fraction of image height to keep (bottom portion)
        self.crop_fraction = rospy.get_param("~crop_fraction", 0.4)
        # Expected lane width as fraction of crop width (used for single-line fallback)
        self.expected_lane_frac = rospy.get_param("~expected_lane_frac", 0.45)
        # EMA smoothing: lower = smoother. Use separate rates for good/poor detection.
        self.ema_alpha_high = rospy.get_param("~ema_alpha_high", 0.35)
        self.ema_alpha_low = rospy.get_param("~ema_alpha_low", 0.10)
        # Morphological kernel size for cleaning masks
        self.morph_size = rospy.get_param("~morph_size", 5)
        # --- sanity-check thresholds -------------------------------------
        # Minimum non-zero pixels to accept a blob as a real lane line
        self.min_blob_pixels = rospy.get_param("~min_blob_pixels", 200)
        # Maximum allowed jump in lane_center (fraction of image width)
        # per frame before we reject the measurement
        self.max_center_jump = rospy.get_param("~max_center_jump", 0.15)
        # How many frames to hold last-good estimate when confidence drops
        self.hold_max = rospy.get_param("~hold_max", 15)

        # --- smoothed state -----------------------------------------------
        self._d_smooth = 0.0
        self._phi_smooth = 0.0
        # Detection confidence: 2 = both lines, 1 = one line, 0 = no lines
        self._confidence = 0
        # Last accepted lane_center in pixel coordinates (for jump detection)
        self._last_center_px = None
        # Hold counter: frames remaining to coast on last-good d/phi
        self._hold_count = 0

        # --- subscriber ---------------------------------------------------
        self.sub_image = rospy.Subscriber(
            "~image/compressed",
            CompressedImage,
            self.cb_image,
            queue_size=1,
            buff_size="20MB",
        )

        # --- publishers ---------------------------------------------------
        self.pub_lane_pose = rospy.Publisher(
            "~lane_pose", LanePose, queue_size=1,
        )
        self.pub_debug_roi = rospy.Publisher(
            "~debug/roi/compressed", CompressedImage, queue_size=1,
        )
        self.pub_debug_mask = rospy.Publisher(
            "~debug/mask/compressed", CompressedImage, queue_size=1,
        )
        self.pub_debug_overlay = rospy.Publisher(
            "~debug/overlay/compressed", CompressedImage, queue_size=1,
        )

        self.log("Lane detector node initialized.")

    # ------------------------------------------------------------------
    def cb_image(self, msg):
        """Receive a camera frame, estimate d and phi, publish LanePose."""
        # Decode compressed image
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            return

        h, w, _ = image.shape

        # Crop to bottom portion (road area only)
        crop_y = int(h * (1.0 - self.crop_fraction))
        crop = image[crop_y:, :]
        ch, cw, _ = crop.shape
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # --- colour masks -------------------------------------------------
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)

        # Morphological open+close to remove noise
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_size, self.morph_size)
        )
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

        # --- raw centroids + pixel counts ---------------------------------
        white_cx, white_area = self._centroid_x_area(white_mask)
        yellow_cx, yellow_area = self._centroid_x_area(yellow_mask)

        # Keep raw values for debug overlay (before rejection)
        raw_white_cx = white_cx
        raw_yellow_cx = yellow_cx

        # --- GATE 1: minimum blob size ------------------------------------
        if white_cx is not None and white_area < self.min_blob_pixels:
            white_cx = None
        if yellow_cx is not None and yellow_area < self.min_blob_pixels:
            yellow_cx = None

        # --- GATE 2: position sanity --------------------------------------
        # Yellow line should be in the left 70% of the image
        if yellow_cx is not None and yellow_cx > cw * 0.70:
            yellow_cx = None
        # White line should be in the right 70% of the image
        if white_cx is not None and white_cx < cw * 0.30:
            white_cx = None
        # If both visible, white must be to the right of yellow
        if white_cx is not None and yellow_cx is not None:
            if white_cx < yellow_cx:
                white_cx = None  # reject the white — more likely to be noise

        # --- top/bottom centroids for phi (only on validated masks) -------
        half = ch // 2
        white_cx_top = self._centroid_x(white_mask[:half, :]) if white_cx is not None else None
        white_cx_bot = self._centroid_x(white_mask[half:, :]) if white_cx is not None else None
        yellow_cx_top = self._centroid_x(yellow_mask[:half, :]) if yellow_cx is not None else None
        yellow_cx_bot = self._centroid_x(yellow_mask[half:, :]) if yellow_cx is not None else None

        # --- estimate lane center -----------------------------------------
        mid = cw / 2.0
        expected_half_lane = cw * self.expected_lane_frac / 2.0
        lane_center = None

        if white_cx is not None and yellow_cx is not None:
            lane_center = (white_cx + yellow_cx) / 2.0
            self._confidence = 2
        elif yellow_cx is not None:
            lane_center = yellow_cx + expected_half_lane
            self._confidence = 1
        elif white_cx is not None:
            lane_center = white_cx - expected_half_lane
            self._confidence = 1
        else:
            self._confidence = 0

        # --- GATE 3: jump detection ---------------------------------------
        # If lane_center jumps too far from last accepted position, reject it
        if lane_center is not None and self._last_center_px is not None:
            jump = abs(lane_center - self._last_center_px) / cw
            if jump > self.max_center_jump:
                # Suspicious jump — likely intersection noise
                self._confidence = 0
                lane_center = None

        # --- hold-last-good fallback or accept new measurement ------------
        if self._confidence >= 1 and lane_center is not None:
            # Good measurement — update last-known and reset hold counter
            self._last_center_px = lane_center
            self._hold_count = self.hold_max
        elif self._hold_count > 0:
            # No good measurement but we still have hold budget → coast
            self._hold_count -= 1
        # else: hold budget exhausted, will decay d/phi toward zero

        holding = (self._confidence == 0 and self._hold_count > 0)

        # --- compute d (lateral offset from image centre) -----------------
        if lane_center is not None:
            d_raw = (lane_center - mid) / mid
        elif holding:
            # Coast: keep last d/phi unchanged (d_raw = current smooth value)
            d_raw = self._d_smooth
        else:
            # No detection AND hold exhausted: decay toward zero
            d_raw = self._d_smooth * 0.8

        # --- compute phi (heading error from top/bottom centroid shift) ---
        phi_raw = 0.0
        if self._confidence == 2:
            top_ok = (white_cx_top is not None and yellow_cx_top is not None)
            bot_ok = (white_cx_bot is not None and yellow_cx_bot is not None)
            if top_ok and bot_ok:
                center_top = (white_cx_top + yellow_cx_top) / 2.0
                center_bot = (white_cx_bot + yellow_cx_bot) / 2.0
                phi_raw = (center_top - center_bot) / cw
            else:
                phi_raw = d_raw * 0.3
        elif self._confidence == 1:
            phi_raw = d_raw * 0.3
        elif holding:
            phi_raw = self._phi_smooth  # coast on last value
        else:
            phi_raw = self._d_smooth * 0.2

        # --- EMA smoothing (faster when confident, slower when unsure) ----
        if holding:
            alpha = 0.0  # freeze smoothed values during hold
        elif self._confidence >= 2:
            alpha = self.ema_alpha_high
        else:
            alpha = self.ema_alpha_low
        self._d_smooth += alpha * (d_raw - self._d_smooth)
        self._phi_smooth += alpha * (phi_raw - self._phi_smooth)

        # --- publish LanePose ---------------------------------------------
        pose = LanePose()
        pose.header.stamp = rospy.Time.now()
        pose.d = self._d_smooth
        pose.phi = self._phi_smooth
        pose.in_lane = (self._confidence >= 1 or holding)
        self.pub_lane_pose.publish(pose)

        # --- debug images (only if someone is listening) ------------------
        self._publish_debug(crop, white_mask, yellow_mask,
                            raw_white_cx, raw_yellow_cx,
                            white_cx, yellow_cx,
                            lane_center, mid, holding, msg.header)

    # ------------------------------------------------------------------
    def _publish_debug(self, crop, white_mask, yellow_mask,
                       raw_white_cx, raw_yellow_cx,
                       white_cx, yellow_cx,
                       lane_center, mid, holding, header):
        """Publish debug images only when there are subscribers."""
        stamp = rospy.Time.now()

        # Debug: ROI crop
        if self.pub_debug_roi.get_num_connections() > 0:
            self.pub_debug_roi.publish(
                self._make_compressed(crop, stamp)
            )

        # Debug: combined mask (white=blue channel, yellow=green channel)
        if self.pub_debug_mask.get_num_connections() > 0:
            mask_vis = np.zeros_like(crop)
            mask_vis[:, :, 0] = white_mask   # blue  = white line
            mask_vis[:, :, 1] = yellow_mask  # green = yellow line
            self.pub_debug_mask.publish(
                self._make_compressed(mask_vis, stamp)
            )

        # Debug: overlay with centroid lines + lane center
        if self.pub_debug_overlay.get_num_connections() > 0:
            overlay = crop.copy()
            ch = overlay.shape[0]
            # draw image centre (grey)
            cv2.line(overlay, (int(mid), 0), (int(mid), ch),
                     (128, 128, 128), 1)
            # draw REJECTED yellow centroid (red, thin) if it was filtered out
            if raw_yellow_cx is not None and yellow_cx is None:
                cv2.line(overlay, (int(raw_yellow_cx), 0), (int(raw_yellow_cx), ch),
                         (0, 0, 255), 1)
                cv2.putText(overlay, "Y?", (int(raw_yellow_cx) + 4, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            # draw REJECTED white centroid (red, thin)
            if raw_white_cx is not None and white_cx is None:
                cv2.line(overlay, (int(raw_white_cx), 0), (int(raw_white_cx), ch),
                         (0, 0, 255), 1)
                cv2.putText(overlay, "W?", (int(raw_white_cx) + 4, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            # draw ACCEPTED yellow line centroid (yellow, solid)
            if yellow_cx is not None:
                cv2.line(overlay, (int(yellow_cx), 0), (int(yellow_cx), ch),
                         (0, 255, 255), 2)
                cv2.putText(overlay, "Y", (int(yellow_cx) + 4, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            # draw ACCEPTED white line centroid (white, solid)
            if white_cx is not None:
                cv2.line(overlay, (int(white_cx), 0), (int(white_cx), ch),
                         (255, 255, 255), 2)
                cv2.putText(overlay, "W", (int(white_cx) + 4, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # draw estimated lane center (green, thick)
            if lane_center is not None:
                cv2.line(overlay, (int(lane_center), 0), (int(lane_center), ch),
                         (0, 255, 0), 2)
                cv2.putText(overlay, "C", (int(lane_center) + 4, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # show smoothed d, phi, confidence, and hold status
            conf_label = ["NONE", "LOW", "HIGH"][self._confidence]
            hold_str = "  HOLD" if holding else ""
            info = f"d={self._d_smooth:.2f}  phi={self._phi_smooth:.2f}  [{conf_label}]{hold_str}"
            cv2.putText(overlay, info, (5, ch - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            self.pub_debug_overlay.publish(
                self._make_compressed(overlay, stamp)
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _make_compressed(image, stamp):
        """Encode a BGR image into a CompressedImage message."""
        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode(".jpg", image)[1]).tobytes()
        return msg

    # ------------------------------------------------------------------
    @staticmethod
    def _centroid_x(mask):
        """Return the x-coordinate of the centroid of non-zero pixels, or None."""
        moments = cv2.moments(mask)
        if moments["m00"] > 0:
            return moments["m10"] / moments["m00"]
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _centroid_x_area(mask):
        """Return (centroid_x, pixel_count) or (None, 0)."""
        moments = cv2.moments(mask)
        area = moments["m00"] / 255.0  # m00 on binary mask counts 255-valued pixels
        if area > 0:
            return moments["m10"] / moments["m00"], area
        return None, 0


if __name__ == "__main__":
    node = LaneDetectorNode(node_name="lane_detector_node")
    rospy.spin()
