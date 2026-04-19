import glob
import time
import os
import sys
sys.path.append(".")
import argparse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from peft import LoraConfig
from diffusers import (
    StableDiffusion3Pipeline,
)
from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny  import  AutoencoderTiny
from thop import profile, clever_format

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="your model", help='path to the pretrained sd3')
    parser.add_argument("--lora_dir", type=str, default="your lora", help='path to tsd-sr lora weights')
    parser.add_argument("--vae_path", type=str, default="your vae", help='path to tsd-sr lora weights')
    
    parser.add_argument("--embedding_dir", type=str, default="dataset/default/", help='path to prompt embeddings')
    
    parser.add_argument("--rank", type=int, default=64, help='rank for transformer')
    parser.add_argument("--rank_vae", type=int, default=64, help='rank for vae')
    
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upscale", type=int, default=4, help='upscale factor')
    parser.add_argument("--process_size", type=int, default=512, help='process size for images')
    parser.add_argument("--mixed_precision", type=str, choices=['fp16', 'fp32'], default="fp16")
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain', help='color alignment method')
        
    return parser.parse_args()

tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])


class TinySR(nn.Module):
    def __init__(self):
        super().__init__()
        transformer = TinySD3Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path,subfolder="transformer", 
                                            torch_dtype=weight_dtype, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
        vae = AutoencoderTiny.from_pretrained(args.vae_path, torch_dtype=weight_dtype)
        self.timesteps = torch.tensor([1000.]).to(device=args.device, dtype=weight_dtype)
        self.vae = vae.to(device=args.device, dtype=weight_dtype)
        self.transformer = transformer.to(args.device, dtype=weight_dtype)
        
        self.prompt_embeds = torch.load(os.path.join(args.embedding_dir, "prompt_embeds.pt"), map_location=args.device).to(dtype=weight_dtype)
        self.pooled_prompt_embeds = torch.load(os.path.join(args.embedding_dir, "pool_embeds.pt"), map_location=args.device).to(dtype=weight_dtype)
        
    def forward(self, pixel_values, size):
        with torch.no_grad():
            # Preprocess the input image
            pixel_values = torch.nn.functional.interpolate(pixel_values, size=size, mode='bicubic', align_corners=False)
            pixel_values = pixel_values * 2 - 1
            pixel_values = pixel_values.to(args.device, dtype=weight_dtype).clamp(-1,1)
            
            # Encode the input image
            model_input = self.vae.encode(pixel_values).latents * self.vae.config.scaling_factor
            model_input = model_input.to(args.device, dtype=weight_dtype)

            # Predict
            model_pred =  self.transformer(
                        hidden_states=model_input,
                        timestep=self.timesteps,
                        pooled_projections=self.pooled_prompt_embeds,
                        return_dict=False,
                    )[0]
            latent_stu = model_input - model_pred
            
            # Decode the output
            image = self.vae.decode(latent_stu / self.vae.config.scaling_factor, return_dict=False)[0].squeeze(0).clamp(-1,1)
        return image


if __name__ == "__main__":
    args = parse_args()
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
        
    model = TinySR()

    input_shape_multi1 = (1, 3, 128, 128) # Batch size 1, 10 features
    num_iterations = 100

    for _ in range(num_iterations):
        dummy_input_multi1 = torch.randn(input_shape_multi1).to(device=args.device, dtype=weight_dtype)
        dummy_input_multi2 = (512, 512)

        model(dummy_input_multi1, dummy_input_multi2)


    param_cnt = sum(p.numel() for p in model.parameters() )
    print("#Param.", param_cnt/1e6, "M")
    
    total_macs = 0
    total_params = 0
    num_iterations = 1
    print(f"Calculating MACs and Parameters for {num_iterations} iterations...")

    for _ in range(num_iterations):
        dummy_input_multi1 = torch.randn(input_shape_multi1).to(device=args.device, dtype=weight_dtype)
        dummy_input_multi2 = (512, 512)

        macs, params = profile(model, inputs=(dummy_input_multi1, dummy_input_multi2), verbose=False)

        total_macs += macs
        total_params += params

    average_macs = total_macs / num_iterations
    average_params = total_params / num_iterations

    average_macs_formatted, average_params_formatted = clever_format([average_macs, average_params], "%.6f")

    print(f"\n--- Average Results over {num_iterations} Iterations ---")
    print(f"Multi-input Model Average MACs: {average_macs_formatted}")
    print(f"Multi-input Model Average Parameters: {average_params_formatted}")

    
            



