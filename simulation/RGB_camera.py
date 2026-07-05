import carla
import numpy as np
import cv2
import time

# Global variable to hold the latest frame as a numpy array (BGR)
latest_frame = None

def image_callback(image):
    global latest_frame
    # Convert raw BGRA bytes to a NumPy array
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    # Convert BGRA to BGR for OpenCV compatibility
    latest_frame = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)

# Set up CARLA client and world
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Camera blueprint and transform
bp_lib = world.get_blueprint_library()
cam_bp = bp_lib.find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x', '600')
cam_bp.set_attribute('image_size_y', '462')
cam_bp.set_attribute('fov', '90')
cam_transform = carla.Transform(
    carla.Location(x=-5.977786, y=-15.654809, z=80.036598),  # Example for center above intersection (adjust as needed)
    carla.Rotation(pitch=-80.055519, yaw=91.906624, roll=-0.000158)
)


camera = world.spawn_actor(cam_bp, cam_transform)
camera.listen(image_callback)

print("Camera started. Capturing frames as NumPy arrays...")

try:
    while True:
        if latest_frame is not None:
            # Now you can process latest_frame as a NumPy array
            cv2.imshow("CARLA Camera Frame", latest_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        time.sleep(0.01)
except KeyboardInterrupt:
    pass
finally:
    camera.stop()
    camera.destroy()
    cv2.destroyAllWindows()
    print("Camera stopped, windows closed.")
