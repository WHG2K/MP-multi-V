import pickle
import pandas as pd

# 读取pkl文件
with open('combined_data_test.pkl', 'rb') as f:
    data = pickle.load(f)

# 保存为Excel
# data.to_excel('combined_data.xlsx', index=False)
print(type(data))
# data.to_json('combined_data_test.json', orient='records', indent=4, force_ascii=False)

# print("数据已保存为 combined_data.xlsx")
