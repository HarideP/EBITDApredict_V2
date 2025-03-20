import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import os
from tqdm import tqdm, trange
from pathlib import Path
import pandas as pd

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
# 设置全局变量控制序列长度和预测长度
SEQ_LENGTH = 20  # 输入序列长度
PRED_LENGTH = 6  # 预测长度
# 设置全局变量控制使用的模型类型：'LSTM'或'GRU'
MODEL_TYPE = 'GRU'  # 可选值: 'LSTM', 'GRU'



# 定义数据集类
class EBITDADataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # 解码最后一个时间步的隐藏状态
        out = self.fc(out[:, -1, :])
        return out

# 定义GRU模型
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播GRU
        out, _ = self.gru(x, h0)
        
        # 解码最后一个时间步的隐藏状态
        out = self.fc(out[:, -1, :])
        return out

# 加载数据
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# 数据预处理
def preprocess_data(data, seq_length=SEQ_LENGTH, pred_length=PRED_LENGTH):
    financial_data = data['financial_data']
    
    # 提取所有特征
    features = []
    for quarter in financial_data:
        feature_dict = {k: v for k, v in quarter.items() if k not in ['date', 'Year']}
        features.append(list(feature_dict.values()))
    
    features = np.array(features)
    
    # 标准化数据
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    # 检查是否有足够的数据进行训练
    available_samples = len(scaled_features) - seq_length - pred_length + 1 - pred_length
    if available_samples <= 0:
        print(f"警告：没有足够的数据进行训练。需要至少 {seq_length + 2*pred_length - 1} 个数据点，但只有 {len(scaled_features)} 个。")
        # 返回空数组
        return np.array([]), np.array([]), np.array([]), np.array([]), scaler
    
    # 创建训练集
    X_train = []
    y_train = []
    
    # 根据要求创建滑动窗口训练数据
    for i in range(available_samples):
        X_train.append(scaled_features[i:i+seq_length])
        # 提取EBITDA值作为预测目标 (假设EBITDA是第一个特征)
        ebitda_indices = np.zeros(len(scaled_features[0]))
        ebitda_indices[0] = 1  # 假设EBITDA是第一个特征
        ebitda_mask = ebitda_indices.astype(bool)
        
        y_values = scaled_features[i+seq_length:i+seq_length+pred_length, ebitda_mask]
        y_train.append(y_values.flatten())
    
    # 创建测试集 (最后一个窗口)
    X_test = [scaled_features[len(scaled_features)-seq_length-pred_length:len(scaled_features)-pred_length]]
    y_test = [scaled_features[len(scaled_features)-pred_length:, 0]]  # 假设EBITDA是第一个特征
    
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), scaler

# 训练模型
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    losses = []
    
    # 使用trange创建带进度条的epoch循环
    epoch_iterator = trange(num_epochs, desc="Training", position=0)
    
    for epoch in epoch_iterator:
        epoch_loss = 0
        # 使用tqdm创建带进度条的batch循环
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, position=1)
        
        for X_batch, y_batch in batch_iterator:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # 前向传播
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            # 更新batch进度条的描述，显示当前损失
            batch_iterator.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        # 更新epoch进度条的描述，显示平均损失
        epoch_iterator.set_postfix(avg_loss=f"{avg_loss:.4f}")
    
    return losses

# 评估模型
def evaluate_model(model, X_test, y_test, scaler, device):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred = model(X_test_tensor).cpu().numpy()
    
    
    # 确保y_pred是二维数组
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(1, -1)  # 将一维数组转为二维
    
    # 反标准化预测结果
    # 为每个预测的季度创建一个单独的反标准化结果
    y_pred_rescaled = []
    for i in range(y_pred.shape[1]):  # 遍历每个预测的季度
        dummy = np.zeros((y_pred.shape[0], scaler.scale_.shape[0]))
        dummy[:, 0] = y_pred[:, i]  # 提取第i个季度的预测值
        # 反标准化
        rescaled = scaler.inverse_transform(dummy)[:, 0]
        y_pred_rescaled.append(rescaled[0])  # 假设只有一个样本
    
    # 反标准化真实值
    y_test_rescaled = []
    for i in range(y_test.shape[1]):  # 遍历每个真实的季度
        dummy = np.zeros((y_test.shape[0], scaler.scale_.shape[0]))
        dummy[:, 0] = y_test[:, i]  # 提取第i个季度的真实值
        # 反标准化
        rescaled = scaler.inverse_transform(dummy)[:, 0]
        y_test_rescaled.append(rescaled[0])  # 假设只有一个样本
    
    # 转换为numpy数组以便计算指标
    y_pred_rescaled = np.array(y_pred_rescaled)
    y_test_rescaled = np.array(y_test_rescaled)
    
    # 计算评估指标
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    rmse = math.sqrt(mse)
    mape = np.mean(np.abs((y_test_rescaled - y_pred_rescaled) / y_test_rescaled)) * 100
    
    results = {
        'predictions': y_pred_rescaled.tolist(),
        'actual': y_test_rescaled.tolist(),
        'mse': mse,
        'rmse': rmse,
        'mape': mape
    }
    
    return results

# 保存结果
def save_results(results, output_path):
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

# 主函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建预测结果目录
    results_dir = Path('results')
    predictions_dir = results_dir / 'predictions'
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有公司的数据文件
    data_dir = Path('data_clean_json')
    company_files = list(data_dir.glob('*_financial_data.json'))
    
    # 创建汇总结果DataFrame
    summary_results = []
    
    # 遍历处理每个公司
    for company_file in tqdm(company_files, desc='Processing companies'):
        company_name = company_file.stem.replace('_financial_data', '')
        print(f"\nProcessing company: {company_name}")
        
        # 加载数据
        data = load_data(str(company_file))
        
        # 预处理数据
        X_train, y_train, X_test, y_test, scaler = preprocess_data(data)
        
        # 检查是否有足够的数据进行训练
        if len(X_train) == 0:
            print(f"没有足够的数据进行训练，跳过 {company_name}")
            continue
        
        # 创建数据加载器
        train_dataset = EBITDADataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
        
        # 模型参数
        input_size = X_train.shape[2]  # 特征数量
        hidden_size = 64
        num_layers = 2
        output_size = y_train.shape[1]  # 预测未来4个季度的EBITDA
        
        # 初始化模型
        if MODEL_TYPE == 'LSTM':
            model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
            model_name = 'LSTM'
        else:  # MODEL_TYPE == 'GRU'
            model = GRUModel(input_size, hidden_size, num_layers, output_size).to(device)
            model_name = 'GRU'
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 训练模型
        num_epochs = 100
        losses = train_model(model, train_loader, criterion, optimizer, num_epochs, device)
        
        # 评估模型
        results = evaluate_model(model, X_test, y_test, scaler, device)
        
        # 打印评估指标
        print(f"MSE: {results['mse']:.4f}")
        print(f"RMSE: {results['rmse']:.4f}")
        print(f"MAPE: {results['mape']:.2f}%")
        
        # 保存预测结果
        save_results(results, str(predictions_dir / f'{company_name}_{model_name}_prediction_results.json'))
        
        # 添加到汇总结果
        summary_results.append({
            'company': company_name,
            'mse': results['mse'],
            'rmse': results['rmse'],
            'mape': results['mape']
        })
    
    # 保存汇总结果
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(str(results_dir / f'{MODEL_TYPE}_summary_results.csv'), index=False)
    print(f"\n所有公司处理完成，汇总结果已保存到 results/{MODEL_TYPE}_summary_results.csv")

if __name__ == "__main__":
    main()

