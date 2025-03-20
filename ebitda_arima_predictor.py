import json
import numpy as np
import pandas as pd
import math
import os
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

# 忽略警告信息
warnings.filterwarnings("ignore")

# 设置随机种子以确保结果可复现
np.random.seed(42)

# 设置全局变量控制序列长度和预测长度
SEQ_LENGTH = 20  # 输入序列长度
PRED_LENGTH = 6  # 预测长度
# 设置全局变量控制使用的模型类型：'ARIMA'或'SARIMA'
MODEL_TYPE = 'SARIMA'  # 可选值: 'ARIMA', 'SARIMA'

# 加载数据
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# 数据预处理
def preprocess_data(data, seq_length=SEQ_LENGTH, pred_length=PRED_LENGTH):
    financial_data = data['financial_data']
    
    # 提取EBITDA数据
    ebitda_data = []
    dates = []
    for quarter in financial_data:
        ebitda_data.append(quarter['ebitda'])
        dates.append(quarter['date'])
    
    ebitda_data = np.array(ebitda_data)
    
    # 检查是否有足够的数据进行训练
    n = len(ebitda_data)
    if n < seq_length + 2 * pred_length:
        print(f"警告：没有足够的数据进行训练。需要至少 {seq_length + 2*pred_length} 个数据点，但只有 {n} 个。")
        # 返回空数组
        return None, None, None, None, None, None, None
    
    # 划分数据集、验证集和测试集
    # 数据集: 0 ~ n-2*pred_length-2
    # 验证集: n-2*pred_length-1 ~ n-pred_length-1
    # 测试集: n-pred_length ~ n-1
    train_data = ebitda_data[:n-2*pred_length-1]
    val_data = ebitda_data[n-2*pred_length-1:n-pred_length]
    test_data = ebitda_data[n-pred_length:]
    
    # 标准化数据
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
    val_data_scaled = scaler.transform(val_data.reshape(-1, 1)).flatten()
    test_data_scaled = scaler.transform(test_data.reshape(-1, 1)).flatten()
    
    return train_data, val_data, test_data, train_data_scaled, val_data_scaled, test_data_scaled, scaler

# 检查时间序列的平稳性
def check_stationarity(timeseries):
    # 进行ADF测试
    result = adfuller(timeseries)
    # 获取p值
    p_value = result[1]
    # 如果p值小于0.05，则认为时间序列是平稳的
    return p_value < 0.05

# 确定ARIMA模型的差分阶数
def determine_d(timeseries, max_d=2):
    for d in range(max_d + 1):
        if d == 0:
            if check_stationarity(timeseries):
                return d
        else:
            diff_series = np.diff(timeseries, n=d)
            if check_stationarity(diff_series):
                return d
    return max_d  # 如果无法确定，则返回最大值

# 确定SARIMA模型的季节性参数
def determine_seasonal_order(timeseries, s=4):
    # 默认季节性周期为4（季度数据）
    # 尝试不同的季节性差分阶数
    best_aic = float('inf')
    best_order = None
    
    for P in range(3):  # 季节性AR阶数
        for D in range(2):  # 季节性差分阶数
            for Q in range(3):  # 季节性MA阶数
                try:
                    model = ARIMA(timeseries, 
                                 order=(1,1,1),  # 临时的非季节性参数
                                 seasonal_order=(P,D,Q,s))
                    model_fit = model.fit()
                    aic = model_fit.aic
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (P,D,Q,s)
                except:
                    continue
    
    return best_order if best_order is not None else (1,1,1,s)  # 如果无法确定，返回默认值

# 训练ARIMA/SARIMA模型
def train_arima_model(train_data, val_data=None):
    # 如果有验证集，则将训练集和验证集合并
    if val_data is not None:
        combined_data = np.concatenate([train_data, val_data])
    else:
        combined_data = train_data
    
    # 确定差分阶数
    d = determine_d(combined_data)
    
    if MODEL_TYPE == 'SARIMA':
        # 确定季节性参数
        seasonal_order = determine_seasonal_order(combined_data)
        print(f"季节性参数: {seasonal_order}")
    
    # 尝试不同的p和q值，选择AIC最小的模型
    best_aic = float('inf')
    best_model = None
    best_params = None
    
    for p in range(6):  # 0-5
        for q in range(6):  # 0-5
            try:
                if MODEL_TYPE == 'ARIMA':
                    model = ARIMA(combined_data, order=(p, d, q))
                else:  # SARIMA
                    model = ARIMA(combined_data, 
                                 order=(p, d, q),
                                 seasonal_order=seasonal_order)
                model_fit = model.fit()
                aic = model_fit.aic
                
                if aic < best_aic:
                    best_aic = aic
                    best_model = model_fit
                    best_params = (p, d, q)
            except:
                continue
    
    print(f"最佳{MODEL_TYPE}参数: {best_params}")
    return best_model

# 使用ARIMA模型进行预测
def predict_with_arima(model, pred_length):
    # 预测未来pred_length个时间点
    forecast = model.forecast(steps=pred_length)
    return forecast

# 评估模型
def evaluate_model(predictions, actual, scaler):
    # 反标准化预测结果和真实值
    predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actual_rescaled = scaler.inverse_transform(actual.reshape(-1, 1)).flatten()
    
    # 计算评估指标
    mse = mean_squared_error(actual_rescaled, predictions_rescaled)
    rmse = math.sqrt(mse)
    mape = np.mean(np.abs((actual_rescaled - predictions_rescaled) / actual_rescaled)) * 100
    
    results = {
        'predictions': predictions_rescaled.tolist(),
        'actual': actual_rescaled.tolist(),
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
    # 创建预测结果目录
    results_dir = Path(f'{MODEL_TYPE}_{SEQ_LENGTH}_{PRED_LENGTH}_results')
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
        train_data, val_data, test_data, train_scaled, val_scaled, test_scaled, scaler = preprocess_data(data)
        
        # 检查是否有足够的数据进行训练
        if train_data is None:
            print(f"没有足够的数据进行训练，跳过 {company_name}")
            continue
        
        try:
            # 训练ARIMA模型
            model = train_arima_model(train_scaled, val_scaled)
            
            # 使用模型进行预测
            predictions = predict_with_arima(model, len(test_scaled))
            
            # 评估模型
            results = evaluate_model(predictions, test_scaled, scaler)
            
            # 打印评估指标
            print(f"MSE: {results['mse']:.4f}")
            print(f"RMSE: {results['rmse']:.4f}")
            print(f"MAPE: {results['mape']:.2f}%")
            
            # 保存预测结果
            save_results(results, str(predictions_dir / f'{company_name}_ARIMA_prediction_results.json'))
            
            # 添加到汇总结果
            summary_results.append({
                'company': company_name,
                'mse': results['mse'],
                'rmse': results['rmse'],
                'mape': results['mape']
            })
        except Exception as e:
            print(f"处理 {company_name} 时出错: {str(e)}")
            continue
    
    # 保存汇总结果
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(str(results_dir / f'{MODEL_TYPE}_summary_results.csv'), index=False)
    print(f"\n所有公司处理完成，汇总结果已保存到 {results_dir}/{MODEL_TYPE}_summary_results.csv")

if __name__ == "__main__":
    main()