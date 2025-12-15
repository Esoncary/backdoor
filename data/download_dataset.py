import os
import json
import pandas as pd
import traceback
from datasets import load_dataset

def download_sst2_for_attngcg():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(current_dir, "sst")
    os.makedirs(target_dir, exist_ok=True)

    try:
        dataset = load_dataset('SetFit/sst2')
        # 定义 AttnGCG 需要的标签映射 (0->Negative, 1->Positive)
        label_map = {0: "Negative", 1: "Positive"}

        def process_and_save_csv(dataset_split, split_name):
            df = pd.DataFrame(dataset_split)
            # 映射 text -> goal (作为攻击的输入上下文)
            if 'text' in df.columns:
                df = df.rename(columns={'text': 'goal'})
            elif 'sentence' in df.columns:
                df = df.rename(columns={'sentence': 'goal'})
                
            # 映射 label -> target (作为攻击的预期输出)
            if 'label' in df.columns:
                df['target'] = df['label'].map(label_map)
            
            # 过滤：只保留 AttnGCG 需要的列
            if 'goal' in df.columns and 'target' in df.columns:
                df = df[['goal', 'target']]
            else:
                print(f"⚠️ 警告: {split_name} 数据集缺少必要列 (goal/target)，跳过处理。现有列: {df.columns}")
                return

            save_path = os.path.join(target_dir, f"{split_name}.csv")
            df.to_csv(save_path, index=False)
        for split_key in dataset.keys():
            process_and_save_csv(dataset[split_key], split_key)

        print(f"\n✨ 所有数据处理完成！现在可以在 main.py 中使用 'data/sst/train.csv' 了。")
        
    except Exception as e:
        print(f"\n❌ 下载或处理过程中出现错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    download_sst2_for_attngcg()