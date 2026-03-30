from mmengine.config.config import ConfigDict
from huggingface_hub import hf_hub_download

# Used to return the dataset class from the config
def instantiate_class_from_config(config):
    if not isinstance(config, ConfigDict) \
        and not isinstance(config, list) \
        and not isinstance(config, dict):
        return config

    if (isinstance(config, ConfigDict) or isinstance(config, dict)) and "target" in config:
        params = config.get("params", {})
        for key, value in params.items():
            params[key] = instantiate_class_from_config(value)
        target_class = config["target"]
        target = target_class(**params)
        return target
    else:
        if isinstance(config, list):
            return [instantiate_class_from_config(c) for c in config]
        elif isinstance(config, ConfigDict) or isinstance(config, dict):
            return {k: instantiate_class_from_config(v) for k, v in config.items()}
        else:
            raise AssertionError(f"Invalid config type: {type(config)}")

import torch
import os


def init_model(config):
    model = instantiate_class_from_config(config['model'])

    if 'pretrained_from' in config and config['pretrained_from'] is not None:
        # Use pretrained_from if we do not want to resume the optimizer/scheduler (just load in weights)
        if not os.path.exists(config['pretrained_from']):
            raise FileNotFoundError(f"Pretrained model not found at {config['pretrained_from']}")
        state_dict = torch.load(config['pretrained_from'], map_location='cpu', weights_only=True)

        load_result = model.load_state_dict(state_dict, strict=False) # change strict to False since we are loading in DAv2

    elif 'resumed_from' in config and config['resumed_from'] is not None:
        # Use resumed_from if we want to load in the optimizer/scheduler state and resume training from a checkpoint (not just load in weights)
        
        if os.path.exists(config['resumed_from']):
            print(f"Loading resumed checkpoint from local path: {config['resumed_from']}")
            checkpoint = torch.load(config['resumed_from'], map_location='cpu', weights_only=True)
        else:
            try:
                ckpt_path = hf_hub_download(
                    repo_id=f"venkatsubra/{config['resumed_from']}", # Access files at the public repo
                    filename="model.pth",
                )
                checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            except Exception as e:
                raise FileNotFoundError(f"Resumed model not found at {config['resumed_from']} locally or on Hugging Face Hub at repo: venkatsubra/{config['resumed_from']}. Error: {str(e)}")
        
        
        model_dict = checkpoint['model']
        new_model_dict = dict()
        for key, value in model_dict.items():
            new_key = key.replace('module.', '')
            new_model_dict[new_key] = value
        # return
        load_result = model.load_state_dict(new_model_dict, strict=False)
        
        if load_result.missing_keys:
            raise RuntimeError(f"Missing keys when loading resumed checkpoint: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            import warnings
            warnings.warn(f"Unexpected keys in resumed checkpoint (ignored): {load_result.unexpected_keys}")
    
    else:
        raise ValueError("Either pretrained_from or resumed_from must be specified in the config to initialize the model with weights for evaluation.")
        
    model.cuda(config['local_rank'])
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config['local_rank']], broadcast_buffers=False,
                                                      output_device=config['local_rank'], find_unused_parameters=True)

    if 'freeze_modules' in config and len(config['freeze_modules']) > 0:
        for name, param in model.named_parameters():
            if any(module in name for module in config['freeze_modules']):
                param.requires_grad = False

    return model