import cv2
from ultralytics import YOLO

# モデルのロード
model = YOLO("yolo11n.pt")

# クラス名リストを取得
class_names = model.names  # dict形式: {0: 'person', 1: 'bicycle', 2: 'car', ...}

def detect_on_frame(frame, conf=0.25):
    # 推論
    results = model.predict(frame, conf=conf, stream=False)

    for res in results:
        boxes = res.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf_score = box.conf[0].item()
            cls_id = int(box.cls[0].item())

            # クラス名を取得
            cls_name = class_names.get(cls_id, str(cls_id))

            # 描画
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, f"{cls_name} {conf_score:.2f}",
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return frame

def run_camera(conf=0.25):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラを開けません")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_on_frame(frame, conf=conf)

        cv2.imshow("YOLO11 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera(conf=0.3)
