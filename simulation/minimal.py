import glob
import os
import sys
import time
import cv2
import numpy as np

# --- 1. CARLA Setup (Standard boilerplate for CARLA scripts) ---
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from ultralytics import YOLO

# --- 2. Configuration Constants (4-Road Quadrants) ---
MODEL_PATH = 'yolov8n.pt' 
CONFIDENCE_THRESHOLD = 0.35 
CENTER_H = 0.50 
CENTER_V = 0.50 
VEHICLE_CLASS_IDS = [2, 3, 5, 7] 

ROAD_LABELS = {
    1: "Road 1 (Top-Left)",
    2: "Road 2 (Top-Right)",
    3: "Road 3 (Bottom-Left)",
    4: "Road 4 (Bottom-Right)"
}

# --- 3. Globals and Callbacks (Unchanged) ---
latest_frame = None

def image_callback(image):
    global latest_frame
    i = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    i = np.reshape(i, (image.height, image.width, 4))[:, :, :3]
    latest_frame = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)

# --- 4. VehicleTracker Class (FIXED for Stable Label Updates) ---

class VehicleTracker:
    def __init__(self, max_missing_frames=10, distance_threshold=50):
        self.tracks = {}
        self.next_id = 0
        self.max_missing_frames = max_missing_frames
        self.distance_threshold = distance_threshold
        self.road_crossings = {1: 0, 2: 0, 3: 0, 4: 0} 
        self.center_x = None
        self.center_y = None

    def _get_clean_class_name(self, class_id, model):
        name = model.names.get(class_id, 'Unknown').upper()
        if name in ['CAR', 'BUS', 'TRUCK', 'MOTORCYCLE']:
            return name
        return 'UNKNOWN'
    
    def _get_road_id(self, x_center, y_center):
        """Determines the road/quadrant ID based on the detection center."""
        if x_center < self.center_x and y_center < self.center_y:
            return 1 # Top-Left
        elif x_center >= self.center_x and y_center < self.center_y:
            return 2 # Top-Right
        elif x_center < self.center_x and y_center >= self.center_y:
            return 3 # Bottom-Left
        elif x_center >= self.center_x and y_center >= self.center_y:
            return 4 # Bottom-Right
        return 0

    def update(self, detections, img_height, img_width, model):
        self.center_y = int(img_height * CENTER_V)
        self.center_x = int(img_width * CENTER_H)
        
        current_associations = {} 

        # 1. Update existing tracks
        for track_id, track_data in list(self.tracks.items()):
            
            min_dist = float('inf')
            best_det_idx = -1
            
            for i, (det_x, det_y, det_cls) in enumerate(detections):
                distance = np.sqrt((det_x - track_data['center'][0])**2 + (det_y - track_data['center'][1])**2)
                
                det_road_id = self._get_road_id(det_x, det_y)

                if distance < min_dist and distance < self.distance_threshold and i not in current_associations and det_road_id == track_data['road_id']:
                    min_dist = distance
                    best_det_idx = i
            
            # 2. Association found
            if best_det_idx != -1:
                det_x, det_y, det_cls = detections[best_det_idx]
                
                # --- FIX FOR STABLE LABEL STARTS HERE ---
                new_label = self._get_clean_class_name(det_cls, model)
                current_stable_label = track_data['stable_label']

                # Update logic:
                # 1. Always update if current label is 'UNKNOWN'.
                # 2. Update if the new label is different and more specific than 'CAR' (e.g., TRUCK or BUS).
                # This ensures a CAR can become a TRUCK, but a TRUCK won't become a CAR.
                if current_stable_label == 'UNKNOWN' or (new_label != current_stable_label and new_label != 'UNKNOWN' and current_stable_label == 'CAR'):
                    track_data['stable_label'] = new_label
                # If the current label is specific (TRUCK/BUS) and the new label is CAR, we ignore the CAR classification.
                
                # --- FIX FOR STABLE LABEL ENDS HERE ---

                # Counting Logic (Unchanged from previous road-counting logic)
                if not track_data.get('has_counted', False):
                    prev_x, prev_y = track_data['center']
                    current_road = track_data['road_id']
                    
                    counted = False
                    if current_road == 1: 
                        if det_x > self.center_x or det_y > self.center_y: counted = True
                    elif current_road == 2: 
                        if det_x < self.center_x or det_y > self.center_y: counted = True
                    elif current_road == 3: 
                        if det_x > self.center_x or det_y < self.center_y: counted = True
                    elif current_road == 4: 
                        if det_x < self.center_x or det_y < self.center_y: counted = True
                            
                    if counted:
                        self.road_crossings[current_road] += 1
                        track_data['has_counted'] = True
                        
                # Update track data
                track_data['center'] = (det_x, det_y)
                track_data['frames_missing'] = 0
                track_data['past_positions'].append((det_x, det_y))

                current_associations[best_det_idx] = track_id

            # 3. No association (Track missing)
            else:
                track_data['frames_missing'] += 1
                if track_data['frames_missing'] > self.max_missing_frames:
                    del self.tracks[track_id]
        
        # 4. Handle new detections (unassociated detections)
        for i, (det_x, det_y, det_cls) in enumerate(detections):
            if i not in current_associations:
                new_id = self.next_id
                
                road_id = self._get_road_id(det_x, det_y) 
                
                if road_id in ROAD_LABELS:
                    self.tracks[new_id] = {
                        'center': (det_x, det_y),
                        'frames_missing': 0,
                        'stable_label': self._get_clean_class_name(det_cls, model),
                        'past_positions': [(det_x, det_y)],
                        'has_counted': False,
                        'road_id': road_id  
                    }
                    current_associations[i] = new_id
                    self.next_id += 1
                
        return current_associations

# --- 5. Main Function (Unchanged display logic) ---

def main():
    global latest_frame, model
    
    # --- CARLA CLIENT SETUP (Unchanged) ---
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    
    # Sensor configuration
    cam_bp = bp_lib.find('sensor.camera.rgb')
    img_w, img_h = 600, 462
    cam_bp.set_attribute('image_size_x', str(img_w))
    cam_bp.set_attribute('image_size_y', str(img_h))
    cam_bp.set_attribute('fov', '90')
    
    # Camera transform 
    cam_transform = carla.Transform(
        carla.Location(x=-79.505730, y=136.943848, z=41.354233), 
        carla.Rotation(pitch=-89.000237, yaw=0.703167, roll=0.000173)
    )
    
    camera = world.spawn_actor(cam_bp, cam_transform)
    camera.listen(image_callback)

    try:
        model = YOLO(MODEL_PATH) 
        print("Camera and YOLO started. Press 'q' to quit.")

    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        if 'camera' in locals() and camera:
            camera.destroy()
        return

    # Initialize the tracker
    tracker = VehicleTracker()

    try:
        while True:
            if latest_frame is not None:
                frame_to_process = latest_frame.copy()
                frame_rgb = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)

                # 1. Run inference
                results = model(frame_rgb, classes=VEHICLE_CLASS_IDS, conf=CONFIDENCE_THRESHOLD, verbose=False) 

                annotated_frame_bgr = frame_to_process.copy() 
                
                img_h_proc, img_w_proc, _ = annotated_frame_bgr.shape
                
                center_y_pixel = int(img_h_proc * CENTER_V)
                center_x_pixel = int(img_w_proc * CENTER_H)

                # 2. Draw Quadrant Split Lines
                # Horizontal Split Line (Green)
                cv2.line(annotated_frame_bgr, (0, center_y_pixel), (img_w_proc, center_y_pixel), (0, 255, 0), 2)
                cv2.putText(annotated_frame_bgr, "HORIZONTAL SPLIT", (10, center_y_pixel - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Vertical Split Line (Blue)
                cv2.line(annotated_frame_bgr, (center_x_pixel, 0), (center_x_pixel, img_h_proc), (255, 0, 0), 2)
                cv2.putText(annotated_frame_bgr, "VERTICAL SPLIT", (center_x_pixel + 5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


                # 3. Process Detections and Update Tracker
                detection_data = [] 
                box_data_list = [] 
                
                if results and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        x_c_norm, y_c_norm, _, _ = box.xywhn[0].cpu().numpy()
                        x_c = int(x_c_norm * img_w_proc)
                        y_c = int(y_c_norm * img_h_proc) 
                        
                        class_id = box.cls.item()
                        
                        detection_data.append((x_c, y_c, class_id))
                        
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        box_data_list.append([x1, y1, x2, y2, class_id])
                        
                        # Draw the center point (Red Dot) for tracking visibility
                        cv2.circle(annotated_frame_bgr, (x_c, y_c), 5, (0, 0, 255), -1) 
                        
                current_associations = tracker.update(detection_data, img_h_proc, img_w_proc, model)
                    
                # 4. Manual Labeling (Using Stable Label and Road ID)
                for idx, box_data in enumerate(box_data_list):
                    if idx in current_associations:
                        track_id = current_associations[idx]
                        
                        if track_id in tracker.tracks:
                            track_info = tracker.tracks[track_id]
                            stable_label = track_info['stable_label']
                            road_id = track_info['road_id'] 
                            
                            x1, y1, x2, y2, _ = box_data

                            # Define color based on the road ID for clarity
                            color_map = {
                                1: (255, 165, 0), # Orange (Top-Left)
                                2: (0, 255, 255), # Yellow (Top-Right)
                                3: (255, 0, 255), # Magenta (Bottom-Left)
                                4: (0, 165, 255)  # Teal (Bottom-Right)
                            }
                            color = color_map.get(road_id, (255, 255, 255))
                            
                            label = f"ID:{track_id} | R:{road_id} | {stable_label}"
                            
                            cv2.rectangle(annotated_frame_bgr, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(annotated_frame_bgr, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 5. Display Road-Specific Counts
                y_offset = 30
                for road_id, count in tracker.road_crossings.items():
                    road_name = ROAD_LABELS.get(road_id, f"Road {road_id}")
                    count_text = f"{road_name} Crossings: {count}"
                    
                    # Position the text clearly in the respective quadrant
                    if road_id == 1: 
                        pos = (10, y_offset)
                    elif road_id == 2: 
                        pos = (center_x_pixel + 10, y_offset)
                    elif road_id == 3: 
                        pos = (10, center_y_pixel + y_offset)
                    elif road_id == 4: 
                        pos = (center_x_pixel + 10, center_y_pixel + y_offset)
                    else:
                        continue

                    cv2.putText(annotated_frame_bgr, count_text, pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    y_offset += 20
                
                # Show the final frame
                cv2.imshow("YOLO Road-Specific Traffic Counting (4 Roads)", annotated_frame_bgr)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            time.sleep(0.01)

    except carla.WorldIsDestroyedError:
        print("CARLA World was destroyed. Exiting.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Cleanup
        if 'camera' in locals() and camera:
            camera.stop()
            camera.destroy()
        cv2.destroyAllWindows()
        print("Cleaned up and exiting.")

if __name__ == "__main__":
    main()