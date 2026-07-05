import paho.mqtt.client as mqtt
import json
import time
from ultralytics import YOLO
import numpy as np
import cv2
import os

# Create directory for saved frames
os.makedirs("saved_frames", exist_ok=True)

# --- Lane assignment based on pixel positions ---
def assign_lane(x, y, img_width=640, img_height=480):
    if 0 <= x <= img_width and 0 <= y < img_height * 0.30:
        return "lane_north"
    elif 0 <= x <= img_width and y > img_height * 0.70:
        return "lane_south"
    elif x < img_width * 0.30 and 0 <= y <= img_height:
        return "lane_west"
    elif x > img_width * 0.70 and 0 <= y <= img_height:
        return "lane_east"
    else:
        return "unknown"

# --- Frame acquisition from CARLA ---
latest_frame = None

def image_callback(image):
    global latest_frame
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    frame = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    latest_frame = frame

def get_frame_from_carla():
    global latest_frame
    if latest_frame is not None:
        return latest_frame.copy()
    else:
        return None

# --- CARLA setup (run this once before while loop) ---
import carla

client_carla = carla.Client('localhost', 2000)
client_carla.set_timeout(10.0)
world = client_carla.get_world()
bp_lib = world.get_blueprint_library()
vehicles = world.get_actors().filter('vehicle.*')
vehicles = world.get_actors().filter('vehicle.*')
if not vehicles:
    print("No vehicles found! Please generate traffic first.")
    exit(1)
vehicle = vehicles[0]  # Choose the first available vehicle
cam_bp = bp_lib.find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x', '640')
cam_bp.set_attribute('image_size_y', '480')
cam_bp.set_attribute('fov', '90')
cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
camera.listen(image_callback)

# --- MQTT setup ---
client = mqtt.Client()
client.connect("localhost", 1883)
topic = "/traffic/intersection/vellore/lanecounts"

# --- YOLO Model ---
model = YOLO('best.pt')

print("Publisher running. Ctrl+C to quit.")

try:
    while True:
        frame = get_frame_from_carla()
        if frame is None:
            print("Waiting for frame...")
            time.sleep(0.05)
            continue

        # -- Save the current frame to disk
        cv2.imwrite(f"saved_frames/frame_{int(time.time() * 1000)}.png", frame)

        img_height, img_width = frame.shape[0:2]
        results = model(frame)

        # Aggregate lane counts
        lane_counts = {"lane_north": 0, "lane_south": 0, "lane_east": 0, "lane_west": 0}
        for box in results[0].boxes:
            xywh = box.xywh.cpu().numpy().reshape(-1)
            x_center, y_center = float(xywh[0]), float(xywh[1])
            lane = assign_lane(x_center, y_center, img_width, img_height)
            if lane in lane_counts:
                lane_counts[lane] += 1

        payload = json.dumps(lane_counts)
        client.publish(topic, payload)
        print(f"Published lane counts: {payload}")

        time.sleep(0.1)  # Adjust for frame rate

except KeyboardInterrupt:
    pass
finally:
    camera.stop()
    camera.destroy()
    client.disconnect()
    print("Camera and MQTT publisher stopped.")