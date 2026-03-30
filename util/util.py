from util.gen_focal_stack import gen_focal_stack
import torch

def get_focal_stack_and_fd_list(
    rgb,
    depth,
    depth_valid_mask,
    K,
    config,
    dataset_sampled_from,
    training,
    dataset_focal_stack=None,
    dataset_fd_list=None,
):
    """
    Determine focal stack and focal distance (FD) list for a sample.

    Modes of operation:
    -------------------
    1. Dataset-provided:
        - Use `dataset_focal_stack` and `dataset_fd_list` directly.

    2. Depth-dependent FD list:
        - Generate FD list from ground-truth depth.
        - Each sample may have a different FD list.
        - Synthesize focal stack accordingly.

    3. Fixed FD list:
        - Use predefined FD list from config.
        - Synthesize focal stack accordingly.

    Args:
    -----
    rgb : torch.Tensor
        Shape: [B, 3, H, W]
        Input RGB image.

    depth : torch.Tensor
        Shape: [B, 1, H, W]
        Ground-truth depth map.

    depth_valid_mask : torch.Tensor
        Shape: [B, 1, H, W]
        Mask indicating valid depth values.

    K : torch.Tensor
        Shape: [B, 3, 3]
        Camera intrinsic matrix.

    config : dict
        Configuration dictionary containing:
            val_loader_config:
                - depth_dependent_fd_list (bool): flag to determine if FD list should be picked based on ground-truth depth values (different for each sample)
                - fd_list (List[float])

    dataset_sampled_from : str
        Name of dataset (used for dataset-specific logic).

    training : bool
        Whether model is in training mode.

    dataset_focal_stack : torch.Tensor, optional
        Shape: [B, N, 3, H, W]
        Precomputed focal stack from dataset.

    dataset_fd_list : torch.Tensor, optional
        Shape: [B, N]
        Precomputed focal distances from dataset.

        Note:
        -----
        Must be provided together with `dataset_focal_stack` if used.

    Returns:
    --------
    focal_stack : torch.Tensor
        Shape: [B, N, 3, H, W]

    fd_list : torch.Tensor
        Shape: [B, N]
    """

    assert training == False, "This function is currently only implemented for evaluation."
    # assert that both dataset_focal_stack and dataset_fd_list are provided together if at all
    if dataset_focal_stack is not None or dataset_fd_list is not None:
        assert dataset_focal_stack is not None and dataset_fd_list is not None, "Both dataset_focal_stack and dataset_fd_list should be provided together if at all"

    if dataset_focal_stack is None and dataset_fd_list is None:
        # Synthetically generate focus stack
        if (not config['val_loader_config'].get('depth_dependent_fd_list', False)) and config['val_loader_config'].get('fd_list') is None:
            raise ValueError(f"fd_list must be specified for {dataset_sampled_from} evaluation in val_config, or depth_dependent_fd_list must be True")

        # Set flags for focal stack generation based on config
        fd_list_params = {
            'depth_dependent': config['val_loader_config'].get('depth_dependent_fd_list', False),
            'fd_list': config['val_loader_config'].get('fd_list', None),
            }

        focal_stack, fd_list, _ = gen_focal_stack(depth, rgb, K, fnumber=config['val_loader_config']['fnumber'], N=config['val_loader_config']['focal_stack_size'], fd_list_params=fd_list_params, psf_type='gauss', p=None) # Only generate focal stack using GaussPSF when evaluation
    else:
        # Pull focal stack and fd list from dataset
        focal_stack, fd_list = dataset_focal_stack, dataset_fd_list
        
    if config['training_with_canonical_depth']:
        # Convert fd_list to canonical fd_list
        width = torch.tensor(focal_stack.shape[-1]).to(depth.device).expand_as(fd_list)
        focal_length = torch.max(K[:,0,0], K[:,1,1]).expand_as(fd_list)  # adjust focal length for resizing
        scaling_factor = (width / focal_length)
        fd_list = fd_list * scaling_factor
    if focal_stack is None or fd_list is None:
        raise ValueError("fd_list or focal_stack is None. Check the focal stack generation step for errors.")

    return focal_stack, fd_list

def run_model_on_sample(model, focal_stack, fd_list, evaluating_model_trained_with_canonical_depth, K):
    """Runs the model on the provided focal_stack and fd_list, handling canonicalization"""

    pd = model(focal_stack, fd_list)

    # During training we keep the predicted depth in canonicalized space if using canonical depth
    if evaluating_model_trained_with_canonical_depth:
        # Revert canonicalization of predicted depth
        width = torch.tensor(focal_stack.shape[-1]).to(focal_stack.device).expand_as(pd)

        focal_length = torch.max(K[:,0,0], K[:,1,1]).expand_as(pd)
        scaling_factor = (width / focal_length)

        pd = pd / scaling_factor
        
        width_fd = torch.tensor(focal_stack.shape[-1]).to(focal_stack.device).expand_as(fd_list)
        focal_length_fd = torch.max(K[:,0,0], K[:,1,1]).expand_as(fd_list)  # adjust focal length for resizing
        fd_scaling_factor = (width_fd / focal_length_fd)
        fd_list = fd_list / fd_scaling_factor

    return pd