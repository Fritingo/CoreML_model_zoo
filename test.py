from ultralytics import YOLO
model = YOLO("yolov10n.pt")
included_class = [0,1]
model.export(format="coreml",imgsz=[640,640],nms=True)