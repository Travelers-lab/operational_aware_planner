import torch
import torch.nn as nn
import numpy as np


class FastLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, output_size=3):
        super(FastLSTM, self).__init__()
        # 使用单层LSTM保持高效
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # 精简的全连接层
        self.fc = nn.Linear(hidden_size, output_size)

        # 初始化参数以加速收敛
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)

    def forward(self, x):
        # 使用CUDA图兼容的简洁结构
        lstm_out, _ = self.lstm(x)
        # 仅使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)


# 输入输出归一化处理器
class DataProcessor:
    def __init__(self):
        # 输入参数范围
        self.input_mins = torch.tensor([1e-5, 1e-5, 0.0])
        self.input_maxs = torch.tensor([1e-3, 1e-1, 20.0])

        # 输出参数范围
        self.output_mins = torch.tensor([0.0, 0.0, 0.0])
        self.output_maxs = torch.tensor([200.0, 100.0, 20.0])

    def normalize_input(self, x):
        """将输入数据归一化到[0,1]范围"""
        # 对数变换处理数量级差异大的前两个特征
        x_log = torch.empty_like(x)
        x_log[:, :, 0] = torch.log10(x[:, :, 0])
        x_log[:, :, 1] = torch.log10(x[:, :, 1])
        x_log[:, :, 2] = x[:, :, 2]

        # 计算归一化参数（基于对数变换后的范围）
        log_mins = torch.tensor([-5.0, -5.0, 0.0])
        log_maxs = torch.tensor([-3.0, -1.0, 20.0])

        return (x_log - log_mins) / (log_maxs - log_mins)

    def denormalize_output(self, y):
        """将输出数据反归一化回原始范围"""
        return y * (self.output_maxs - self.output_mins) + self.output_mins


# 示例使用
if __name__ == "__main__":
    # 配置参数
    SEQ_LENGTH = 10  # 输入序列长度
    BATCH_SIZE = 1  # 批处理大小

    # 初始化模型和处理器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastLSTM().to(device).eval()  # 设置为评估模式
    processor = DataProcessor()

    # 生成模拟输入数据 (1x10x3)
    sample_input = torch.tensor([
        [np.random.uniform(1e-5, 1e-3),
         np.random.uniform(1e-5, 1e-1),
         np.random.uniform(0, 20)]
        for _ in range(SEQ_LENGTH)
    ]).unsqueeze(0).float().to(device)

    # 归一化输入
    norm_input = processor.normalize_input(sample_input)

    # 使用Torch编译优化模型 (PyTorch 2.0+特性)
    compiled_model = torch.compile(model, mode="max-autotune")

    # 预热GPU
    for _ in range(10):
        _ = compiled_model(norm_input)

    # 性能测试
    import time

    test_cycles = 500
    start_time = time.time()

    for _ in range(test_cycles):
        with torch.no_grad(), torch.cuda.amp.autocast():
            pred = compiled_model(norm_input)

    # 计算频率
    duration = time.time() - start_time
    freq = test_cycles / duration
    print(f"推理频率: {freq:.2f} Hz")

    # 反归一化输出
    final_output = processor.denormalize_output(pred.cpu())
    print("预测输出:", final_output.squeeze().tolist())