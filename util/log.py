import os
import wandb
import logging
import torch.distributed as dist
from termcolor import colored
import numpy as np
import io
import matplotlib.pyplot as plt
from PIL import Image
import torch
import matplotlib.cm as cm
import matplotlib.pyplot as plt


logs = set()

class _SuppressImageSizeWarning(logging.Filter):
    def filter(self, record):
        return 'Images sizes do not match' not in record.getMessage()

logging.getLogger().addFilter(_SuppressImageSizeWarning())

class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log

def setup_logger(log_dir, rank, save_to_file=True, color=True, abbrev_name=None):
    logger = logging.getLogger(f'train_logger_{rank}')
    logger.handlers.clear()  # Clear any existing handlers
    logger.propagate = False
    logger.setLevel(logging.INFO)
    
    plain_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    if rank == 0:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        if color:
            if abbrev_name is None:
                abbrev_name = ""
            color_formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=logger.name,
                abbrev_name=abbrev_name
            )
            stream_handler.setFormatter(color_formatter)
        else:
            stream_handler.setFormatter(plain_formatter)
        logger.addHandler(stream_handler)
    
    if save_to_file:
        file_handler = logging.FileHandler(os.path.join(log_dir, f'rank{rank}.log'))
        file_handler.setFormatter(plain_formatter)
        logger.addHandler(file_handler)
    
    return logger

def make_config_serializable(config_obj):
    if isinstance(config_obj, dict):
        return {k: make_config_serializable(v) for k, v in config_obj.items()}
    elif isinstance(config_obj, list):
        return [make_config_serializable(item) for item in config_obj]
    elif isinstance(config_obj, (int, float, str, bool)) or config_obj is None:
        return config_obj
    else:
        return str(config_obj)

def init_wandb(config):
    safe_config = make_config_serializable(config._cfg_dict)
    wandb.init(
        project=config['project_name'],
        dir=config['base_log_dir'],
        name=config['experiment_name'],
        config=safe_config,
    )
    config['log_dir'] = wandb.run.dir
    config.dump(os.path.join(config['log_dir'], 'config.py'))

def broadcast_wandb_dir(config):
    wandb_dir_list = [config.get('log_dir', None)]
    dist.broadcast_object_list(wandb_dir_list, src=0)
    config['log_dir'] = wandb_dir_list[0]

def wandb_log_scalars(scalars: dict, step: int, split='train'):
    for name, value in scalars.items():
        wandb.log({f'{split}/{name}': value}, step=step)

def wandb_log_images(images: dict, colorbar_ticks, depth_normalizer, step: int, split='train', index=0, depth_valid_mask=None, dataset_name='unknown'):
    images_list = []

    for name, image in images.items():
        assert isinstance(image, np.ndarray)

        if name in ('pd', 'gt'):
            # Make a copy to avoid modifying the original
            disp_image = image.copy()

            if depth_valid_mask is not None and name == 'gt':
                # Wherever depth_valid_mask == 0, set depth to NaN so colormap ignores it
                disp_image = np.where(depth_valid_mask == 0, np.nan, disp_image)

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(disp_image, cmap='Spectral', norm=depth_normalizer, interpolation='nearest')
            ax.set_title(f'Depth Map ({name})')
            cbar = plt.colorbar(im, ax=ax, ticks=colorbar_ticks)
            cbar.ax.set_yticklabels([f"{tick:.2f}" for tick in colorbar_ticks])


            buf = io.BytesIO()
            plt.tight_layout()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            plt.close(fig)

            image = np.array(Image.open(buf))
        images_list.append(wandb.Image(image, caption=name))
    
    # Log difference between predicted and ground truth depth if both are available
    if 'pd' in images and 'gt' in images:
        # (pd-gt)/std(gt)
        disp_pd_image = images['pd'].copy()
        disp_pd_image = np.where(depth_valid_mask == 0, np.nan, disp_pd_image) if depth_valid_mask is not None else disp_pd_image
        disp_gt_image = images['gt'].copy()
        disp_gt_image = np.where(depth_valid_mask == 0, np.nan, disp_gt_image) if depth_valid_mask is not None else disp_gt_image
        diff = np.clip(disp_pd_image - disp_gt_image, -2, 2)
        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.get_cmap("bwr").copy()
        cmap.set_bad(color="0.2")         # NaNs show as dark gray
        im = ax.imshow(diff, cmap=cmap, vmin=-2, vmax=2, interpolation='nearest')
        ax.set_title('Depth Difference (pd - gt)')
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.set_yticklabels([f"{tick:.2f}" for tick in cbar.get_ticks()])

        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        plt.close(fig)

        diff_image = np.array(Image.open(buf))
        images_list.append(wandb.Image(diff_image, caption='depth_difference'))



        # Matplotlib-independent logging
        pred_norm = depth_normalizer(images['pd'])
        gt_norm   = depth_normalizer(images['gt'])

        pred_rgb = cm.get_cmap("Spectral")(pred_norm)[..., :3]
        gt_rgb   = cm.get_cmap("Spectral")(gt_norm)[..., :3]

        pred_uint8 = (pred_rgb * 255).astype(np.uint8)
        gt_uint8   = (gt_rgb   * 255).astype(np.uint8)

        diff = np.clip(images['pd'] - images['gt'], -2, 2)
        invalid = np.isnan(diff) | (~depth_valid_mask if depth_valid_mask is not None else False)
        diff_norm = (diff + 2) / 4.0
        diff_rgb = cm.get_cmap("bwr")(diff_norm)[..., :3]
        diff_rgb[invalid] = 0.2
        diff_uint8 = (diff_rgb * 255).astype(np.uint8)

        # Append into SAME panel (same key)
        images_list.append(wandb.Image(pred_uint8, caption="pred"))
        images_list.append(wandb.Image(gt_uint8, caption="gt"))
        images_list.append(wandb.Image(diff_uint8, caption="diff"))

        wandb.log({f'{split}/{dataset_name}/sample_{index}': images_list}, step=step)

def wandb_log_focal_stack(focal_stack, step, fd_list, split='train', index=0, dataset_name='unknown'):
    """
    Log a focal stack (as a grid and/or as individual images) to wandb with a unique key.
    Args:
        focal_stack: torch.Tensor [N, 3, H, W] -- no batch dimension
        step: int
        fd_list: list of focal distances corresponding to the focal planes in the stack
        split: str
        index: int, unique sample index
    """

    if isinstance(fd_list, torch.Tensor):
        fd_list = fd_list.cpu().numpy().tolist()

    imgs = []
    for i in range(focal_stack.shape[0]):
        img = focal_stack[i].cpu().permute(1, 2, 0).numpy()


        def denormalize_image(image):
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            return image * std + mean
        img_array = denormalize_image(img)*255
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)


        caption = f"Focal plane {i}, fd={fd_list[i]:.3f}"
        imgs.append(wandb.Image(img_array, caption=caption))


    wandb.log({
        f"{split}/{dataset_name}/focal_stack_planes_{index}": imgs
    }, step=step)

def wandb_log_coc_map(coc_map, step, fd_list, split='train', index=0, dataset_name='unknown'):
    """
    Log a focal stack (as a grid and/or as individual images) to wandb with a unique key.
    Args:
        focal_stack: torch.Tensor [N, 3, H, W] or [B, N, 3, H, W]
        step: int
        split: str
        index: int, unique sample index
    """
    # If batch dimension exists, select the first sample
    cmap = 'Spectral'

    if coc_map.dim() == 4:
        coc_map = coc_map[0]  # [N, H, W]

    N = coc_map.shape[0]
    ncols = 3
    nrows = (N + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = axes.flatten()

    for i, fd in enumerate(fd_list):
        ax = axes[i]


        coc_max = 20  # Cap the maximum CoC value
        coc_normalizer = plt.Normalize(vmin=0, vmax=coc_max)
        coc_vis = ax.imshow(np.clip(coc_map[i].cpu().numpy(), 0, coc_max), cmap=cmap, norm=coc_normalizer)
        ax.set_title(f'Circle of Confusion (fd={fd})')
        fig.colorbar(coc_vis, ax=ax)
    # Hide any unused subplots
    for i in range(N, len(axes)):
        axes[i].axis('off')


    # Need to use the buffer to save the figure
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=150)
    buf.seek(0)
    plt.close(fig)

    # Convert buffer to PIL Image
    pil_img = Image.open(buf)
    
    # Log the CoC map to wandb
    wandb.log({
        f"{split}/{dataset_name}/coc_map_grid_{index}": wandb.Image(pil_img, caption=f"CoC Map Grid {split} idx {index}")
    }, step=step)
    