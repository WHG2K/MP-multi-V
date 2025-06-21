import pickle
import sys
import os
import pandas as pd

# 指定目录路径
directory = "old_data/Experiment_MP"

print(f"正在读取目录: {directory}")

# 用户选择模式
print("请选择模式:")
print("1. 全部保存")
print("2. 轻量版测试（每个文件只保存前10个数据）")

try:
    choice = input("请输入选择 (1 或 2): ")
    if choice == "2":
        test_mode = True
        print("选择轻量版测试模式")
    else:
        test_mode = False
        print("选择全部保存模式")
except:
    test_mode = False
    print("默认选择全部保存模式")

# 存储所有数据的列表
all_data = []

# 遍历目录中的所有pickle文件
for filename in os.listdir(directory):
    if filename.endswith('.pkl'):
        file_path = os.path.join(directory, filename)
        print(f"正在读取: {filename}")
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # 如果数据是字典，转换为DataFrame
            if isinstance(data, dict):
                df = pd.DataFrame(data)
                # 添加文件名列以便追踪
                df['source_file'] = filename
            elif isinstance(data, pd.DataFrame):
                data['source_file'] = filename
                df = data
            else:
                print(f"跳过 {filename} - 不是字典或DataFrame格式")
                continue
            
            # 只删除sim_data列，保留menu列
            if 'sim_data' in df.columns:
                df = df.drop(columns=['sim_data'])
            
            # 如果是测试模式，只取前10行
            if test_mode and len(df) > 10:
                df = df.head(10)
                print(f"  测试模式：只取前10行数据")
            
            all_data.append(df)
                
        except Exception as e:
            print(f"读取 {filename} 时发生错误: {e}")

if all_data:
    # 合并所有DataFrame
    print("正在合并所有数据...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"合并后的数据形状: {combined_df.shape}")
    print(f"列名: {list(combined_df.columns)}")
    
    # 删除source_file列
    if 'source_file' in combined_df.columns:
        combined_df = combined_df.drop(columns=['source_file'])
        print("已删除source_file列")
    
    # 重命名列
    column_mapping = {
        'opt_max_x': 'bf_x',
        'K': 'B'
    }
    combined_df = combined_df.rename(columns=column_mapping)
    print("已重命名列: opt_max_x -> bf_x, K -> B")
    
    # 删除opt_rev_in_sample列
    if 'opt_rev_in_sample' in combined_df.columns:
        combined_df = combined_df.drop(columns=['opt_rev_in_sample'])
        print("已删除opt_rev_in_sample列")
    
    # 按照指定顺序重新排列列
    desired_order = ['menu', 'N', 'B', 'C', 'F', 'N_0', 'u', 'r', 'v', 'bf_x']
    
    # 检查哪些列存在
    existing_columns = [col for col in desired_order if col in combined_df.columns]
    other_columns = [col for col in combined_df.columns if col not in desired_order]
    
    # 重新排列列顺序
    final_columns = existing_columns + other_columns
    combined_df = combined_df[final_columns]
    
    print(f"已重新排列列顺序")
    print(f"前10列: {final_columns[:10]}")
    
    print(f"最终数据形状: {combined_df.shape}")
    print(f"最终列名: {list(combined_df.columns)}")
    
    # 显示前几行数据
    print("\n前5行数据:")
    print(combined_df.head())
    
    # 保存合并后的数据为JSON格式
    if test_mode:
        output_file = "combined_data_test.jsonl"
    else:
        output_file = "combined_data.jsonl"
    
    # combined_df.to_json(output_file, orient='records', indent=4, force_ascii=False)
    combined_df.to_json(
        output_file,
        orient='records',
        lines=True,
        force_ascii=False,
    )
    print(f"\n数据已保存到: {output_file}")
    
else:
    print("没有找到可用的pickle文件")
