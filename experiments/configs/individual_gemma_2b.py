import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.tokenizer_paths=["/home/dataset/2024_zox_llm/code/AttnGCG/model/AI-ModelScope/gemma-2b-it"]
    config.model_paths=["/home/dataset/2024_zox_llm/code/AttnGCG/model/AI-ModelScope/gemma-2b-it"] 
    config.conversation_templates=['gemma']
    config.devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
    
    config.attention_weight = 50.0

    return config