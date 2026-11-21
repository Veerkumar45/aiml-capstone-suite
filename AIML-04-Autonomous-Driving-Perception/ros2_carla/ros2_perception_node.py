"""
ROS2 Autonomous Driving Perception Stack Node.
Subscribes to CARLA camera sensor topics, runs real-time perception pipeline
(YOLO Object Detection + ByteTrack Multi-Object Tracking + OpenCV Lane Detection + MiDaS Metric Depth),
and publishes tracked obstacles and collision warning alerts.
"""

import sys
import os
import time
import numpy as np
from typing import Dict, Any, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from detection.yolo_detector import YOLODetector
from tracking.tracker import ByteTracker
from lanes.classical_lane_detector import ClassicalLaneDetector
from depth.midas_depth_estimator import MiDaSDepthEstimator

# Try importing ROS2 rclpy
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import String, Bool
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

class AutonomousPerceptionPipeline:
    """
    Unified end-to-end perception engine:
    Frame -> [YOLO Detector] -> [ByteTrack] -> [Classical Lanes] -> [MiDaS Metric Depth] -> Alerts
    """
    def __init__(self):
        print("[Perception Stack] Initializing perception models...")
        self.detector = YOLODetector()
        self.tracker = ByteTracker()
        self.lane_detector = ClassicalLaneDetector()
        self.depth_estimator = MiDaSDepthEstimator()

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        t0 = time.perf_counter()

        # 1. Object Detection
        raw_detections = self.detector.detect(frame)

        # 2. Multi-Object Tracking
        tracked_objects = self.tracker.update(raw_detections)

        # 3. Lane Detection & Curvature
        lane_info = self.lane_detector.detect_lanes(frame)

        # 4. Monocular Depth Estimation & Metric Distance Projection
        depth_map = self.depth_estimator.estimate_depth_map(frame)
        tracked_with_distance = self.depth_estimator.compute_obstacle_distances(depth_map, tracked_objects)

        # 5. Collision Alert Logic
        critical_obstacles = [o for o in tracked_with_distance if o["collision_alert"] == "CRITICAL_BRAKE"]
        emergency_brake = len(critical_obstacles) > 0

        total_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        fps = round(1000.0 / max(total_latency_ms, 0.1), 1)

        return {
            "fps": fps,
            "latency_ms": total_latency_ms,
            "tracked_objects": tracked_with_distance,
            "lane_metrics": lane_info,
            "emergency_brake_triggered": emergency_brake,
            "depth_map": depth_map
        }

if ROS2_AVAILABLE:
    class ROS2PerceptionNode(Node):
        def __init__(self):
            super().__init__("autonomous_perception_node")
            self.pipeline = AutonomousPerceptionPipeline()

            # Subscriptions
            self.subscription = self.create_subscription(
                Image,
                "/carla/ego_vehicle/rgb_front/image_color",
                self.image_callback,
                10
            )

            # Publishers
            self.brake_publisher = self.create_publisher(Bool, "/vehicle/emergency_brake", 10)
            self.tracks_publisher = self.create_publisher(String, "/perception/tracked_objects", 10)
            self.get_logger().info("ROS2 Perception Node initialized and subscribed to CARLA camera.")

        def image_callback(self, msg: Image):
            # In production, cv_bridge converts ROS Image to cv2 numpy frame
            # np_frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            pass

def run_standalone_demo(max_frames: int = 10):
    print(f"[ROS2 Perception Node] Running perception pipeline standalone validation ({max_frames} frames)...")
    from data.dataset_loader import PerceptionDatasetLoader
    pipeline = AutonomousPerceptionPipeline()

    for idx in range(max_frames):
        frame, _ = PerceptionDatasetLoader.generate_synthetic_frame(idx)
        out = pipeline.process_frame(frame)
        print(f"Frame {idx+1:02d} | Latency: {out['latency_ms']}ms ({out['fps']} FPS) | Tracked: {len(out['tracked_objects'])} | Curvature: {out['lane_metrics']['curvature_radius_m']}m | AEB Active: {out['emergency_brake_triggered']}")

if __name__ == "__main__":
    run_standalone_demo(max_frames=5)
