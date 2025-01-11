from ultralytics import YOLO
import cv2
import cvzone
import os
import json

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("../Yolo-weights/yolov8n.pt")

classNames = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

subObjectClasses = {
    "person": ["cell phone", "laptop", "backpack", "umbrella", "handbag", "tie", "suitcase", "book", "teddy bear", "helmet"]
}

os.makedirs("subobject_images", exist_ok=True)
os.makedirs("output_json", exist_ok=True)

output_results = []

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    detected_objects = []
    frame_data = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            cvzone.cornerRect(img, (x1, y1, w, h), colorR=(255, 0, 255))

            conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])
            obj_name = classNames[cls]

            cvzone.putTextRect(img, f'{obj_name} {conf}', (max(0, x1), max(35, y1)))

            detected_objects.append({
                "object": obj_name,
                "bbox": [x1, y1, x2, y2],
                "confidence": conf
            })

    for obj in detected_objects:
        obj_name = obj["object"]
        x1, y1, x2, y2 = obj["bbox"]

        if obj_name in subObjectClasses:
            parent_data = {
                "object": obj_name,
                "bbox": obj["bbox"],
                "confidence": obj["confidence"],
                "subobject": []
            }

            for sub_obj in detected_objects:
                sub_obj_name = sub_obj["object"]
                sub_x1, sub_y1, sub_x2, sub_y2 = sub_obj["bbox"]

                if sub_obj_name in subObjectClasses[obj_name] and x1 <= sub_x1 and y1 <= sub_y1 and x2 >= sub_x2 and y2 >= sub_y2:
                    parent_data["subobject"].append({
                        "object": sub_obj_name,
                        "bbox": sub_obj["bbox"],
                        "confidence": sub_obj["confidence"]
                    })

            frame_data.append(parent_data)

        elif obj_name not in [parent["object"] for parent in frame_data]:
            frame_data.append({
                "object": obj_name,
                "bbox": obj["bbox"],
                "confidence": obj["confidence"]
            })

    output_results.append(frame_data)

    cv2.imshow("Detection", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if cv2.getWindowProperty("Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

json_output_path = os.path.join("output_json", "output_results.json")
try:
    with open(json_output_path, "w") as json_file:
        json.dump(output_results, json_file, indent=4)
    print(f"JSON saved successfully at: {json_output_path}")
except Exception as e:
    print(f"Error saving JSON: {e}")

cap.release()
cv2.destroyAllWindows()


