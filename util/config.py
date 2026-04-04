import argparse
from mmengine import Config
from dataset import DDFF12Loader_Val, Uniformat, InfinigenDefocus, Zedd, HAMMER
from huggingface_hub import snapshot_download, hf_hub_download
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import zipfile

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"   # disables hf_transfer/xet

def get_config(config=None):
    parser = argparse.ArgumentParser(description="Run Depth model.")
    parser.add_argument("--config", type=str, default=config, help="Path to the config file.")
    args, unknown = parser.parse_known_args()

    config = Config.fromfile(args.config)
    return config

def update_hugging_face_dataset_folder(config):
    if config['dataset_location'] is None:
        return  # No dataset location specified (this must be a local dataset), so nothing to update
    local_path = resolve_dataset_folder(config['dataset_location'])
    if config['val_dataset'] == 'DIODE':
        config['dataset']['val']['params']['args'].update({
            "dir_data_indoor": str(local_path / "diode_indoor_v2"),
            "dir_data_outdoor": str(local_path / "diode_outdoor_v2"),
        })
    elif config['val_dataset'] == 'iBims':
        config['dataset']['val']['params']['args'].update({
            "dir_data": str(local_path),
        })
    elif config['val_dataset'] == 'InfinigenDefocus':
        config['dataset']['val']['params']['args'].update({
            "dataset_folder": str(local_path),
        })
    elif config['val_dataset'] == 'Zedd':
        config['dataset']['val']['params']['args'].update({
            "dataset_folder": str(local_path),
        })

import json

def download_and_extract_zip(dataset_cfg):

    zip_path = snapshot_download(
        repo_id=dataset_cfg["repo_id"],
        repo_type="dataset",
        allow_patterns=[dataset_cfg["zip_filename"]],
        local_dir=dataset_cfg["local_dir"],
    )
    print(f"Downloaded zip file for dataset {dataset_cfg['repo_id']} to {zip_path}")
    with zipfile.ZipFile(zip_path + "/" + dataset_cfg["zip_filename"], "r") as zf:
        members = zf.namelist()

        # Detect common top-level folder
        top_levels = set(m.split('/')[0] for m in members if m.strip())
        
        if len(top_levels) == 1:
            top = list(top_levels)[0]
        else:
            top = None  # no single outer folder

        for member in members:
            if member.endswith("/"):
                continue  # skip directories

            # Remove top-level folder if it exists
            if top and member.startswith(top + "/"):
                relative_path = member[len(top) + 1:]
            else:
                relative_path = member

            if not relative_path:
                continue

            target_path = os.path.join(dataset_cfg['local_dir'], relative_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            with zf.open(member) as src, open(target_path, "wb") as dst:
                dst.write(src.read())

    zip_file_path = zip_path + "/" + dataset_cfg["zip_filename"]
    os.remove(zip_file_path)
    print(f"Removed zip file {zip_file_path}")


def resolve_dataset_folder(dataset_cfg):
    if dataset_cfg.get("source") == "huggingface":
        local_dir = Path(dataset_cfg["local_dir"])
        if local_dir.exists():
            print(f"Dataset already exists at {local_dir}, skipping download.")
        else:
            local_dir.mkdir(parents=True, exist_ok=True)
            download_and_extract_zip(dataset_cfg)
        root = local_dir

        print(f"Downloaded dataset {dataset_cfg['repo_id']} to {root}")
        return Path(root) / Path(dataset_cfg["subdir"])
    return Path(dataset_cfg["path"])
