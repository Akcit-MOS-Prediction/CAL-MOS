from typing import List

import torch.nn as nn
from peft import LoraConfig, get_peft_model


WEIGHTS_KEYS_DICT = {
    "key": ["k_proj"],
    "query": ["q_proj"],
    "value": ["v_proj"],
    "linear": ["intermediate_dense", "output_dense"],
    "whisper_linear": ["fc1", "fc2"],
}


def build_target_modules(lora_keys: List[str]) -> List[str]:
    """
    Builds a list of target modules that will be used for the PEFT model.

    Args:
        lora_keys: List of keys that will be used for the PEFT model, e.g. ["key", "query", "value", "linear"]
    Returns:
        List of target modules, e.g. ["k_proj", "q_proj", "v_proj", "intermediate_dense", "output_dense"]
    """
    target_modules = []
    for lora_key in lora_keys:
        try:
            target_modules.extend(WEIGHTS_KEYS_DICT[lora_key])
        except KeyError:
            raise ValueError(f"Invalid lora key: {lora_key}.")

    return target_modules


def build_peft_model(
    model: nn.Module,
    lora_keys: List[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    bias: str,
):
    print("Using PEFT model!!!")
    target_modules = build_target_modules(lora_keys)
    print(f"Target modules: {target_modules}")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
    )

    model = get_peft_model(model, lora_config)
    print(model.print_trainable_parameters())

    return model