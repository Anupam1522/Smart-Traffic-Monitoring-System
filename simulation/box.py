#!/usr/bin/env python

import carla
import random
import time

# ---------------------------------------------
# Enhanced classifier for CARLA vehicles
# ---------------------------------------------
def get_vehicle_category(type_id):
    type_id = type_id.lower()

    # Bikes / Motorcycles
    if any(x in type_id for x in [
        'harley', 'yamaha', 'ninja', 'kawasaki', 'ducati', 'motorcycle', 'bike'
    ]):
        return 'Bike'

    # Bicycles
    if any(x in type_id for x in [
        'bicycle', 'crossbike', 'century', 'bh', 'diamondback'
    ]):
        return 'Bicycle'

    # Trucks & Pickups
    if any(x in type_id for x in [
        'truck', 'pickup', 'carlacola', 'firetruck', 'man.cargotruck'
    ]):
        return 'Truck'

    # Vans
    if any(x in type_id for x in [
        'van', 'volkswagen.t2', 'ford.ambulance'
    ]):
        return 'Van'

    # Buses / Coaches
    if any(x in type_id for x in [
        'bus', 'coach', 'mercedes.sprinter', 'volvo.bus'
    ]):
        return 'Bus'

    # Cars — includes most sedans, SUVs, coupes, hatchbacks
    if any(x in type_id for x in [
        'audi', 'tesla', 'lincoln', 'ford.mustang', 'mini', 'bmw', 'chevrolet',
        'mercedes', 'toyota', 'jeep', 'nissan', 'coupe', 'sedan', 'hatchback',
        'wagon', 'car', 'kia', 'seat', 'carlamotors', 'model3', 'prius'
    ]):
        return 'Car'

    # Fallback
    return 'Vehicle'

# ---------------------------------------------
# Main logic
# ---------------------------------------------
def main():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        # Spawn ego vehicle
        bp = random.choice(blueprint_library.filter('vehicle.*'))
        transform = random.choice(world.get_map().get_spawn_points())
        vehicle = world.try_spawn_actor(bp, transform)
        if vehicle is None:
            print("❌ Could not spawn vehicle. Try again.")
            return

        vehicle.set_autopilot(True)
        print(f"✅ Ego vehicle spawned: {vehicle.type_id}")

        # Debug visualization loop
        while True:
            vehicles = world.get_actors().filter('vehicle.*')

            for v in vehicles:
                bbox = v.bounding_box
                transform = v.get_transform()

                # Determine type
                v_type = get_vehicle_category(v.type_id)

                # Colors for categories
                color_map = {
                    'Car': carla.Color(0, 255, 0),       # Green
                    'Truck': carla.Color(255, 0, 0),     # Red
                    'Bus': carla.Color(255, 165, 0),     # Orange
                    'Bike': carla.Color(0, 255, 255),    # Cyan
                    'Bicycle': carla.Color(255, 255, 0), # Yellow
                    'Van': carla.Color(255, 0, 255),     # Magenta
                    'Vehicle': carla.Color(255, 255, 255)
                }
                color = color_map.get(v_type, carla.Color(255, 255, 255))

                # Draw 3D bounding box
                world.debug.draw_box(
                    bbox,
                    transform.rotation,
                    thickness=0.2,
                    color=color,
                    life_time=0.1
                )

                # Draw text above box
                loc = transform.location + carla.Location(z=2.5)
                world.debug.draw_string(
                    loc,
                    v_type,
                    color=color,
                    life_time=0.1
                )

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n🛑 Script stopped by user.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
