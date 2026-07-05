import carla
import os
import time

output_dir = 'datasets/raw_frames'
os.makedirs(output_dir, exist_ok=True)

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Optionally, load Town03 to ensure map with more vehicles/spawns
world = client.load_world('Town03')

bp_lib = world.get_blueprint_library()

vehicles = world.get_actors().filter('vehicle.*')
if len(vehicles) == 0:
    print("No vehicles found! Please generate traffic first.")
    exit(1)  # Exit if no vehicles are present

vehicle = vehicles[0]  # First available vehicle

# Camera settings
cam_bp = bp_lib.find('sensor.camera.rgb')
cam_bp.set_attribute("image_size_x", "640")
cam_bp.set_attribute("image_size_y", "480")
cam_bp.set_attribute("fov", "90")

# OPTION 1: Camera attached on top of the first vehicle (dynamic)
cam_transform = carla.Transform(
    carla.Location(x=-5.977786, y=-15.654809, z=80.036598),  # Example for center above intersection (adjust as needed)
    carla.Rotation(pitch=-80.055519, yaw=91.906624, roll=-0.000158)
)

camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)

# OPTION 2: Static camera in the world (example coordinates)
# Uncomment below and comment out OPTION 1 to use a fixed world camera
# cam_transform = carla.Transform(carla.Location(x=230, y=195, z=50),
#                                 carla.Rotation(pitch=-30))
# camera = world.spawn_actor(cam_bp, cam_transform)

def save_image(image):
    image.save_to_disk(f"{output_dir}/frame_{image.frame:06d}.png")
    print(f"Saved frame {image.frame}")

camera.listen(save_image)
print("Camera started. Press Ctrl+C to quit.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    camera.stop()
    camera.destroy()
    print("Camera stopped.")