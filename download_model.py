# source /etc/network_turbo
from modelscope import snapshot_download

# 指定模型的下载路径
cache_dir = '/home/dataset/2024_zox_llm/code/AttnGCG/model'
# 调用 snapshot_download 函数下载模型
model_dir = snapshot_download('AI-ModelScope/gemma-2b-it', cache_dir=cache_dir)

print(f"模型已下载到: {model_dir}")