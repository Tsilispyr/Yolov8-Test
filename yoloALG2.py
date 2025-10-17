import cv2
import argparse
import numpy as np
from ultralytics import YOLO
import supervision as sv

ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Load YOLOv8 model
    model = YOLO("yolov8l.pt")

    # Annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator()
    
    # Polygon zone
    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon)
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.RED,
        thickness=2
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        result = model(frame, agnostic_nms=True)[0]

        # Convert to Supervision detections
        detections = sv.Detections.from_ultralytics(result)

        # Prepare labels
        labels = [
            f"{model.model.names[class_id]} {conf:0.2f}"
            for class_id, conf in zip(detections.class_id, detections.confidence)
        ]

        # Annotate boxes and labels
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        # Trigger polygon zone
        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)

        # Display
        cv2.imshow("YOLOv8 Live", frame)

        if cv2.waitKey(30) == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
