import os
import logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from dataset import init_dataloader
from util.config import get_config, update_hugging_face_dataset_folder
from util.log import init_wandb, broadcast_wandb_dir, setup_logger
from util.dist import setup_distributed
from util.init import init_model
from util.metric import MetricTracker
from util.val import validate, measure_model_efficiency

def main(): 
    config = get_config('config/val.py')
    update_hugging_face_dataset_folder(config) # If the validation dataset is from Hugging Face, download it and update the config to point to the downloaded dataset folder
    
    setup_distributed()
    cudnn.enabled = True
    cudnn.benchmark = True

    local_rank = int(os.environ["LOCAL_RANK"])
    config['local_rank'] = local_rank
    print(f"config bs: {config['bs']}, local_rank: {local_rank}")
    assert config['bs'] == 1, "Batch size for validation must be 1"
    assert local_rank == 0, "Validation currently only supports single GPU"

    init_wandb(config)
    broadcast_wandb_dir(config)
    logger = setup_logger(config['log_dir'], local_rank)

    load_model_for_validation(config, logger)

def load_model_for_validation(config, logger):
    model = init_model(config)

    config['val_loader_config'] = config['val_loader_config_options'][config['val_loader_config_choice']]

    config['dataset']['val']['params']['args'].update({'val_loader_config': config['val_loader_config']}) # Necessary for InfinigenDefocus to get validation config options before it is instantiated

    val_loader, val_subset = init_dataloader(config, 'val')


    step = 0

    logger.info(f'Validating for {len(val_loader)} steps, {config["bs"] * torch.cuda.device_count()} samples per step')

    results = validate(model, config, val_loader, val_subset, step, first_epoch=True)
    print("Validation results:", results)

    if config['show_efficiency']:
        efficiency = measure_model_efficiency(model, config, val_loader)
        print("Efficiency metrics:", efficiency)


if __name__ == '__main__':
    main()