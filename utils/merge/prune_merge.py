from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    AutoencoderKL,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline
)
import sys
sys.path.append(".")
import torch
from peft import LoraConfig,LoraModel
import os
from peft import PeftModel
from collections import OrderedDict
from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from utils.util import load_lora_state_dict_warn

pretrained_model_name_or_path = "/your/teacher"
base_dir = "/your/root_dir"
output_dir = os.path.join(base_dir, f"/save/path")
config ="/your/config"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
vae_dir = os.path.join(output_dir, "vae")   
if not os.path.exists(vae_dir):
    os.makedirs(vae_dir)
transformer_dir = os.path.join(output_dir, "transformer")
if not os.path.exists(transformer_dir):
    os.makedirs(transformer_dir)
scheduler_dir = os.path.join(output_dir, "scheduler")
if not os.path.exists(scheduler_dir):
    os.makedirs(scheduler_dir)

# prune_mask = [your mask] 
prune_mask = []

weight_type = torch.float16
noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
transformer = TinySD3Transformer2DModel.from_pretrained(pretrained_model_name_or_path,subfolder="transformer")
vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")

def copy_weight(transformer, transformer_prune, prune_mask):
    prune_index = [i for i,v in enumerate(prune_mask) if v == 0]
    left_index = [i for i,v in enumerate(prune_mask) if v == 1]
            
    left_dict = {key:value for key,value in zip(left_index,range(len(left_index)))}
    print(left_dict)
    parameters = transformer.state_dict()
    copy_parameter = OrderedDict()
    for key,value in parameters.items():
        if "transformer_blocks" in key:
            index = int(key.split(".")[1])
            if index in prune_index:
                continue
            else:
                tmp = left_dict[index]
                key_tmp = key.replace(f"transformer_blocks.{index}.",f"transformer_blocks.{tmp}.")
                copy_parameter[key_tmp] = value
        else:
            copy_parameter[key] = value
    transformer_prune.load_state_dict(copy_parameter)
    return transformer_prune
transformer_prune = TinySD3Transformer2DModel.from_config(config)
transformer_prune = copy_weight(transformer, transformer_prune, prune_mask)

vae.to(weight_type)
transformer_prune.to(weight_type)
print(transformer_prune)

noise_scheduler.save_pretrained(scheduler_dir)
vae.save_pretrained(vae_dir)
transformer_prune.save_pretrained(transformer_dir)





