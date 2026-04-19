# Copyright 2024 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import itertools
from typing import Any, Dict, List, Optional, Union

import sys
sys.path.append(".")

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, RMSNorm
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from models.tinysr.tinysd3block import JointTransformerBlock


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding with support for SD3 cropping."""

    def __init__(
        self,
        height=224,
        width=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=1,
        pos_embed_type="sincos",
        pos_embed_max_size=None,  # For SD3 cropping
    ):
        super().__init__()

        self.flatten = flatten
        self.layer_norm = layer_norm
        self.pos_embed_max_size = pos_embed_max_size

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

        self.patch_size = patch_size
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size
        self.interpolation_scale = interpolation_scale

        print("loading pos_embed")
        self.register_buffer("pos_embed", torch.load("dataset/cache/pos_embed.pt"))
        # self.pos_embed = torch.load("dataset/cache/pos_embed.pt")


    def forward(self, latent):
        latent = self.proj(latent)
        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if self.layer_norm:
            latent = self.norm(latent)
        return (latent + self.pos_embed).to(latent.dtype)


class AdaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        # NOTE: It is a bit weird that the norm layer can be configured to have scale and shift parameters
        # because the output is immediately scaled and shifted by the projected conditioning embeddings.
        # Note that AdaLayerNorm does not let the norm layer have scale and shift parameters.
        # However, this is how it was implemented in the original code, and it's rather likely you should
        # set `elementwise_affine` to False.
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps, elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x, scale, shift

class TinySD3Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
    ):
        super().__init__()
        default_out_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = PatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,  # hard-code for now.
        )
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )

        self.transformer_blocks = nn.ModuleList(
            [
                JointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.inner_dim,
                    context_pre_only=i == num_layers - 1,
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.norm = self.norm_out.norm
        self.register_buffer("scale", torch.tensor([]))
        self.register_buffer("shift", torch.tensor([]))
        
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False
        
        self.initialized = False

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        height, width = hidden_states.shape[-2:]
        
        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.

        if self.initialized is False:
            temb = self.time_text_embed(timestep, pooled_projections)
            del self.time_text_embed
            print("del self.time_text_embed")
            self.temb = temb
        
        for block in self.transformer_blocks:
            if self.initialized is False:
                hidden_states = block(
                    hidden_states=hidden_states, temb=self.temb
                )
            else:
                hidden_states = block.forward_(
                    hidden_states=hidden_states,
                )
        
        if self.initialized is False:
            hidden_states, scale, shift  = self.norm_out(hidden_states, self.temb)
            self.scale = scale
            self.shift = shift
            self.initialized = True
            del self.norm_out
            del self.temb
            print("del self.norm_out")
            print("del self.temb")
        else:
            hidden_states = self.norm(hidden_states) * (1 + self.scale)[:, None, :] + self.shift[:, None, :]
            
        hidden_states = self.proj_out(hidden_states)
        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
    
    def merge_and_unload(
        self, progressbar=False, safe_merge=False, adapter_names=None
    ):

        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )
        
    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names = None,
    ):
        from tqdm import tqdm
        from peft.tuners.tuners_utils import onload_layer
        from peft.utils import _get_submodules, ModulesToSaveWrapper
        key_list = [key for key, _ in self.named_modules() if "lora_" not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "base_layer"):
                    if merge:
                        target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                    self._replace_module(parent, target_name, target.get_base_layer(), target)
                elif isinstance(target, ModulesToSaveWrapper):
                    # save any additional trainable modules part of `modules_to_save`
                    new_module = target.modules_to_save[target.active_adapter]
                    if hasattr(new_module, "base_layer"):
                        # check if the module is itself a tuner layer
                        if merge:
                            new_module.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                        new_module = new_module.get_base_layer()
                    setattr(parent, target_name, new_module)

        return self

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if ("lora_" in name) or ("ranknum" in name):
                weight = child.qweight if hasattr(child, "qweight") else child.weight
                module.to(weight.device)

if __name__ == "__main__":
    model = TinySD3Transformer2DModel.from_pretrained("checkpoint/tinyfusion/tinymerge/prune-12-merge-tinysr", subfolder="transformer", low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
    print(model)
    model = model.to(torch.float16).cuda()
    # torch.save(model.state_dict(), "pytorch_model.bin") # 1.3GB
    a = torch.randn(1, 16, 64, 64).to(torch.float16).cuda()
    b = torch.randn(1, 2048).to(torch.float16).cuda()
    model(a, pooled_projections = b, timestep=torch.tensor([1000]).long().cuda(), return_dict=True)
    print(model)
    