from ultralytics import YOLO
import cv2

def infer_image(model_path, image_path, output_path="output/predictions"):
    """Run inference on a single image and save result."""
    model = YOLO(model_path)
    results = model.predict(image_path, conf=0.5)
    img = cv2.imread(image_path)
    for box in results[0].boxes:
        x, y, w, h = map(int, box.xywh[0])
        cls = int(box.cls[0])
        label = "person" if cls == 0 else "pet"
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    os.makedirs(output_path, exist_ok=True)
    cv2.imwrite(os.path.join(output_path, os.path.basename(image_path)), img)
    return results

if __name__ == "__main__":
    infer_image("models/yolov8n.pt", "test_image.jpg")