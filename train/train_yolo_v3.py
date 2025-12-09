from ultralytics import YOLO

def main():
    model = YOLO("pre-trained_checkpoints/yolov3u.pt")

    model.train(
        data="dataset/voc_2007_dataset_config.yaml",
        epochs=50,
        imgsz=416,
        batch=16,
        lr0=1e-3,
        optimizer="SGD",
        momentum=0.937,
        weight_decay=5e-4,
        device=0,     # or "cpu"
        workers=4,
        project="runs/voc",
        name="yolov3_voc2007"
    )

if __name__ == "__main__":
    main()
