from ultralytics import YOLO
import coremltools as ct
import torch

from coremltools.converters.mil.mil import types

# Load a pretrained YOLOv10n model
# traced_model = YOLO('yolov8n.pt')

# traced_model.export(format="torchscript", nms=True)
# traced_model.export(format="onnx", nms=True)

example_input = torch.rand(1, 3, 640, 640) 
# input1 = ct.TensorType(name='input_ids', shape=example_input.size(), dtype=types.int64)


traced_model = torch.jit.load('yolov8n.torchscript')

# # print(traced_model)

coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape, dtype=types.fp32)],
    source="pytorch",
    convert_to="neuralnetwork"  # 指定轉換格式為 neuralnetwork
)

# # 7. 保存轉換後的 CoreML 模型為 .mlmodel 格式
coreml_model.save("yolov8n.mlmodel")

# # Perform object detection on an image
# results = traced_model("image.jpeg")

# # Display the results
# results[0].show()
