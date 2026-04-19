import torch
import os
from collections import OrderedDict
def load_lora_state_dict(state_dict, model, adapter_name="default"):
    for n, p in model.named_parameters():
        if adapter_name in n:
            name = "transformer." + n.replace(f".{adapter_name}", "")
            p.data.copy_(state_dict[name])
            state_dict.pop(name)
    if len(state_dict) > 0:
        print(f"Warning: {len(state_dict)} keys not loaded")
        print(state_dict.keys())
        
def load_lora_state_dict_warn(state_dict, model, adapter_name="default"):
    for n, p in model.named_parameters():
        if adapter_name in n:
            name = "transformer." + n.replace(f".{adapter_name}", "") 
            if name not in state_dict:
                print(f"Warning: {name} not found in state_dict")
                continue
            p.data.copy_(state_dict[name])
            state_dict.pop(name)
    if len(state_dict) > 0:
        print(f"Warning: {len(state_dict)} keys not loaded")
        print(state_dict.keys())
        
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if param.requires_grad:
            # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def select_next_optimized_random_max(gumbel_gate: torch.Tensor, all_options: torch.Tensor, current_opt: torch.Tensor) -> torch.Tensor:
    if current_opt.dim() == 1:
        current_opt = current_opt.unsqueeze(0) 

    intersection_mask_raw = (all_options * current_opt).sum(dim=-1) > 0

    not_self_mask = ~(current_opt.eq(all_options).all(dim=-1))

    final_valid_mask = intersection_mask_raw & not_self_mask

    if not final_valid_mask.any():
        return torch.tensor([]) 

    gumbel_gate = gumbel_gate.squeeze()  # Ensure gumbel_gate is 1D
    filtered_gumbel_gate = gumbel_gate[final_valid_mask]
    filtered_options = all_options[final_valid_mask]

    max_prob = torch.max(filtered_gumbel_gate)

    indices_of_max_prob = (filtered_gumbel_gate == max_prob).nonzero(as_tuple=True)[0]

    random_selection_idx = torch.randint(0, len(indices_of_max_prob), (1,)).item()
    
    best_idx_in_filtered = indices_of_max_prob[random_selection_idx]

    return filtered_options[best_idx_in_filtered]

def teacher_opt_gen_selective_grad(mask: torch.Tensor, next_mask: torch.Tensor, intersection_first: bool = True) -> torch.Tensor:
    mask = mask.float()
    
    next_mask_detached = next_mask.float().detach()

    mask_intersection = mask * next_mask_detached

    mask_union = (mask + next_mask_detached).clamp(0, 1)

    if intersection_first:
        return torch.cat([mask_intersection, mask, mask_union], dim=0)
    else:
        return torch.cat([mask_union, mask, mask_intersection], dim=0)
        