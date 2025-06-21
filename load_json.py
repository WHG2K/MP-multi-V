import json
from tqdm import tqdm

filename = 'combined_data_test.jsonl'

with open(filename, 'r', encoding='utf-8') as f:
    total_instances = sum(1 for _ in f)

with open(filename, 'r', encoding='utf-8') as f:

    # 使用 for 循环逐行遍历文件
    for line in tqdm(f, total=total_instances, desc="Redo SP"):
        
        # line_string 现在是一个字符串，例如: '{"student_id":101,...}\n'
        
        # --- 这是获取字典的关键步骤 ---
        # 使用 json.loads() 将单行字符串解析成一个Python字典
        inst = json.loads(line)

        tqdm.write(f"  成功获取字典: {inst}")

            
        break
        # print("-" * 30)