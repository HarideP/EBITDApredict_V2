import os
import json
import pandas as pd

def get_average_mape(folder_path):
    predictions_dir = os.path.join(folder_path, 'predictions')
    if not os.path.exists(predictions_dir):
        return None
    
    mape_values = []
    for file in os.listdir(predictions_dir):
        if file.endswith('_prediction_results.json'):
            file_path = os.path.join(predictions_dir, file)
            with open(file_path, 'r') as f:
                data = json.load(f)
                mape = data.get('mape')
                if mape and mape < 800:  # 排除MAPE大于800的异常值
                    mape_values.append(mape)
    
    return sum(mape_values) / len(mape_values) if mape_values else None

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_folders = [
        'ARIMA_20_4_results',
        'ARIMA_20_6_results',
        'SARIMA_20_4_results',
        'SARIMA_20_6_results',
        'GRU_20_4_results',
        'GRU_20_6_results',
        'LSTM_20_4_results',
        'LSTM_20_6_results'
    ]
    
    results = []
    for folder in model_folders:
        folder_path = os.path.join(base_dir, folder)
        avg_mape = get_average_mape(folder_path)
        if avg_mape is not None:
            results.append({
                'Model': folder,
                'Average MAPE': round(avg_mape, 2)
            })
    
    # 将结果转换为DataFrame并打印
    df = pd.DataFrame(results)
    print('\n模型预测结果分析（排除MAPE > 800的异常值）：')
    print('=' * 50)
    print(df.to_string(index=False))
    
    # 保存原始结果为CSV
    df.to_csv('model_results.csv', index=False)
    
    # 添加新的分组表格
    # 从Model列中提取模型类型和序列长度
    df['Model_Type'] = df['Model'].apply(lambda x: x.split('_')[0])
    df['Sequence'] = df['Model'].apply(lambda x: '_'.join(x.split('_')[1:3]))
    
    # 创建透视表
    pivot_df = df.pivot(index='Model_Type', columns='Sequence', values='Average MAPE')
    
    # 打印分组结果
    print('\n\n按模型类型和序列长度分组的平均MAPE值：')
    print('=' * 50)
    print(pivot_df.to_string())
    
    # 保存透视表为CSV
    pivot_df.to_csv('model_results_pivot.csv')

if __name__ == '__main__':
    main()