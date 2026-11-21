"""
CARLA Autonomous Driving Simulator Sensor Client.
Spawns ego-vehicle in CARLA simulation, attaches RGB camera sensor,
and streams synchronized video frames at 30 FPS to the ROS2 perception stack.
"""

import time
import numpy as np
from typing import Callable, Optional

# Try importing CARLA client API
try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False

class CarlaCameraStreamer:
    def __init__(self, host: str = "127.0.0.1", port: int = 2000, image_width: int = 1280, image_height: int = 720):
        self.host = host
        self.port = port
        self.width = image_width
        self.height = image_height
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.is_running = False

    def connect_and_spawn(self):
        if not CARLA_AVAILABLE:
            print("[CARLA Notice] carla Python library not available on system. Using high-fidelity synthetic camera generator.")
            return

        try:
            print(f"[CARLA] Connecting to CARLA Simulator at {self.host}:{self.port}...")
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(5.0)
            self.world = self.client.get_world()

            # Blueprint library
            bp_lib = self.world.get_blueprint_library()
            vehicle_bp = bp_lib.filter("model3")[0]

            # Spawn vehicle at map waypoint
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_pt = spawn_points[0] if spawn_points else carla.Transform()
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_pt)
            print(f"[CARLA] Spawned ego vehicle (ID: {self.vehicle.id})")

            # Camera Sensor Blueprint
            cam_bp = bp_lib.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(self.width))
            cam_bp.set_attribute("image_size_y", str(self.height))
            cam_bp.set_attribute("fov", "90")
            cam_bp.set_attribute("sensor_tick", "0.033") # ~30 FPS

            # Mount camera on hood/windshield (x=1.5m forward, z=1.4m height)
            cam_transform = carla.Transform(carla.Location(x=1.5, z=1.4))
            self.camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
            print("[CARLA] Attached front RGB camera sensor.")

        except Exception as e:
            print(f"[CARLA Error] Could not connect to CARLA simulator: {e}")
            self.camera = None

    def start_streaming(self, frame_callback: Callable[[np.ndarray], None], max_frames: int = 100):
        self.is_running = True
        print("[CARLA Streamer] Sensor stream started...")

        if self.camera is not None and CARLA_AVAILABLE:
            def _on_carla_frame(image):
                if not self.is_running:
                    return
                # Convert raw CARLA BGRA to BGR numpy array
                array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (image.height, image.width, 4))
                bgr_frame = array[:, :, :3]
                frame_callback(bgr_frame)

            self.camera.listen(_on_carla_frame)
        else:
            # Standalone Simulator replay for testing
            from data.dataset_loader import PerceptionDatasetLoader
            for i in range(max_frames):
                if not self.is_running:
                    break
                frame, _ = PerceptionDatasetLoader.generate_synthetic_frame(i, width=self.width, height=self.height)
                frame_callback(frame)
                time.sleep(0.033) # 30 FPS timing

    def stop(self):
        self.is_running = False
        if self.camera is not None:
            self.camera.stop()
            self.camera.destroy()
        if self.vehicle is not None:
            self.vehicle.destroy()
        print("[CARLA] Streamer stopped and actors cleaned up.")

if __name__ == "__main__":
    streamer = CarlaCameraStreamer(image_width=640, image_height=480)
    count = [0]
    def dummy_cb(f):
        count[0] += 1
        if count[0] % 10 == 0:
            print(f"[CARLA Callback] Received frame {count[0]} - shape: {f.shape}")
    streamer.connect_and_spawn()
    streamer.start_streaming(dummy_cb, max_frames=20)
    streamer.stop()
