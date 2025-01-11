# Real-Time Object Detection with Parent-Child Relationship Detection

## Overview
This project implements a real-time object detection system using YOLOv8 that not only detects objects but also identifies hierarchical relationships between objects (parent-child relationships). For example, it can detect a person (parent) holding a cell phone (child object) and establish their relationship in the spatial context.

## Features
- Real-time object detection using YOLOv8
- Parent-child relationship detection between objects
- Bounding box visualization with corner styling
- Confidence score display
- JSON output generation for detected objects and their relationships
- Support for multiple object classes and sub-object relationships

## Prerequisites
- Python 3.x
- OpenCV (cv2)
- Ultralytics YOLO
- cvzone
- YOLO weights file (yolov8n.pt)

## Installation
```bash
pip install ultralytics opencv-python cvzone
```

## Project Structure
```
project/
│
├── Yolo-weights/
│   └── yolov8n.pt
│
├── subobject_images/    # Directory for saving detected object images
├── output_json/         # Directory for JSON output files
└── main.py             # Main detection script
```

## Code Components

### 1. Initialization
- Video capture setup (1280x720 resolution)
- YOLO model loading
- Directory creation for outputs

### 2. Object Classes
- Supports 80 standard YOLO classes including:
  - Common objects (person, car, bicycle, etc.)
  - Animals (bird, cat, dog, etc.)
  - Personal items (backpack, umbrella, phone, etc.)
  - Furniture and appliances
  - Food items

### 3. Parent-Child Relationships
```python
subObjectClasses = {
    "person": ["cell phone", "laptop", "backpack", "umbrella", "handbag", 
               "tie", "suitcase", "book", "teddy bear", "helmet"]
    # Expandable for more relationships
}
```

### 4. Detection Pipeline
1. **Frame Capture**: Continuous video frame capture from camera
2. **Object Detection**: YOLO model processes each frame
3. **Bounding Box Creation**: Draws boxes around detected objects
4. **Relationship Analysis**: Checks for parent-child spatial relationships
5. **Visualization**: Displays detection results with:
   - Colored bounding boxes
   - Object labels
   - Confidence scores

### 5. Data Structure
The code generates a hierarchical JSON structure:
```json
[
    {
        "object": "parent_object",
        "bbox": [x1, y1, x2, y2],
        "confidence": confidence_score,
        "subobject": [
            {
                "object": "child_object",
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence_score
            }
        ]
    }
]
```

## Usage
1. Ensure your webcam is connected
2. Run the script:
```bash
python main.py
```
3. Press 'q' to quit the application

## Output
- **Real-time Display**: Shows video feed with detected objects
- **JSON Output**: Generates structured data in `output_json/output_results.json`

## Customization
1. **Adding New Parent-Child Relationships**
   ```python
   subObjectClasses = {
       "existing_parent": ["sub_object1", "sub_object2"],
       "new_parent": ["new_sub_object1", "new_sub_object2"]
   }
   ```

2. **Modifying Resolution**
   ```python
   cap.set(3, 1280)  # Width
   cap.set(4, 720)   # Height
   ```

3. **Changing YOLO Model**
   ```python
   model = YOLO("path_to_different_weights.pt")
   ```

## Performance Considerations
- The script runs in real-time but performance depends on:
  - Hardware capabilities
  - Video resolution
  - Number of objects being tracked
  - Complexity of parent-child relationships

## Limitations
- Requires good lighting conditions for optimal detection
- Parent-child relationship detection based on spatial overlap only
- Fixed set of predefined object classes
- Single camera input support


## License
[MIT License](https://opensource.org/licenses/MIT)
