import torch
import coremltools as ct

# 1. 定義和加載你的 PyTorch 模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1)
        self.fc1 = torch.nn.Linear(16*222*222, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = x.view(-1, 16*222*222)
        x = self.fc1(x)
        return x

# 假設已經有一個訓練好的模型
model = SimpleModel()
model.eval()

# 2. 創建一個示例輸入張量
dummy_input = torch.randn(1, 3, 224, 224)

# 3. 將 PyTorch 模型導出為 TorchScript 格式
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("model.pt")

# 4. 使用 coremltools 將 TorchScript 模型轉換為 CoreML 模型
traced_model = torch.jit.load("model.pt")

# 5. 定義模型的輸入
example_input = torch.randn(1, 3, 224, 224)  # 根據你的模型輸入尺寸進行調整

# 6. 將 TorchScript 模型轉換為 CoreML 模型並保存為 .mlmodel 格式
coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape)],
    convert_to="neuralnetwork"  # 指定轉換格式為 neuralnetwork
)

# 7. 保存轉換後的 CoreML 模型為 .mlmodel 格式
coreml_model.save("model.mlmodel")

