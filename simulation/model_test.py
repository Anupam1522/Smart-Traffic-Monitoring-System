from ultralytics import YOLO
import cv2

# Load your YOLO model (adjust path if needed)
model = YOLO('best.pt')

# Read the saved frame image from your system
image_path = 'frame_077121.png'  # Replace with path if different
img = cv2.imread(image_path)

# Run YOLO detection on the image
results = model(img)

print(f"Number of detections: {len(results[0].boxes)}")

# Print detected class IDs and confidence scores to understand what is detected
for box in results[0].boxes:
    class_id = int(box.cls.cpu().numpy()[0])
    conf = float(box.conf.cpu().numpy()[0])
    print(f"Class ID: {class_id}, Confidence: {conf:.2f}")

    # Optionally draw bounding box to visualize detection
    xywh = box.xywh.cpu().numpy().reshape(-1)
    x_c, y_c, bw, bh = xywh
    x1 = int(x_c - bw/2)
    y1 = int(y_c - bh/2)
    x2 = int(x_c + bw/2)
    y2 = int(y_c + bh/2)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(img, f"ID:{class_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

# Show the image with detections
cv2.imshow("YOLO Detection Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
