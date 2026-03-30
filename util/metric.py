import torch
import numpy as np
from typing import List, Tuple

class MetricTracker:
    def __init__(self):
        self.metrics = {}

    def update(self, metrics):
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    def get_metrics(self):
        return {key: sum(value) / len(value) for key, value in self.metrics.items()}


# From DepthPro: https://github.com/apple/ml-depth-pro/blob/9efe5c1def37a26c5367a71df664b18e1306c708/src/depth_pro/eval/boundary_metrics.py#L192
def fgbg_depth(
    d: np.ndarray, t: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find foreground-background relations between neighboring pixels.

    Args:
    ----
        d (np.ndarray): Depth matrix.
        t (float): Threshold for comparison.

    Returns:
    -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Four matrices indicating
        left, top, right, and bottom foreground-background relations.

    """
    with np.errstate(divide='ignore', invalid='ignore'):
        right_is_big_enough = (d[..., :, 1:] / d[..., :, :-1]) > t
        left_is_big_enough = (d[..., :, :-1] / d[..., :, 1:]) > t
        bottom_is_big_enough = (d[..., 1:, :] / d[..., :-1, :]) > t
        top_is_big_enough = (d[..., :-1, :] / d[..., 1:, :]) > t
    return (
        left_is_big_enough,
        top_is_big_enough,
        right_is_big_enough,
        bottom_is_big_enough,
    )

# VS: added edge mask that only considers boundaries (horizontal or vertical pixel pairs) where both pixels are valid
def edge_masks(mask: np.ndarray):
    """Build per-edge masks so only pairs where BOTH pixels are valid are kept."""
    mh = mask[..., :, 1:] & mask[..., :, :-1]   # for left/right edges (H, W-1)
    mv = mask[..., 1:, :] & mask[..., :-1, :]   # for top/bottom edges (H-1, W)
    return mh, mv

def boundary_f1(
    pr: np.ndarray,
    gt: np.ndarray,
    t: float,
    return_p: bool = False,
    return_r: bool = False,
    mask: np.ndarray = None,
) -> float:
    """Calculate Boundary F1 score.

    Args:
    ----
        pr (np.ndarray): Predicted depth matrix.
        gt (np.ndarray): Ground truth depth matrix.
        t (float): Threshold for comparison.
        return_p (bool, optional): If True, return precision. Defaults to False.
        return_r (bool, optional): If True, return recall. Defaults to False.

    Returns:
    -------
        float: Boundary F1 score, or precision, or recall depending on the flags.

    """
    ap, bp, cp, dp = fgbg_depth(pr, t)
    ag, bg, cg, dg = fgbg_depth(gt, t)

    mh, mv = edge_masks(mask)
    

    # VS: apply the edge masks to only consider valid pixel pairs
    ap, bp, cp, dp = ap & mh, bp & mv, cp & mh, dp & mv
    ag, bg, cg, dg = ag & mh, bg & mv, cg & mh, dg & mv

    r = 0.25 * (
        np.count_nonzero(ap & ag) / max(np.count_nonzero(ag), 1)
        + np.count_nonzero(bp & bg) / max(np.count_nonzero(bg), 1)
        + np.count_nonzero(cp & cg) / max(np.count_nonzero(cg), 1)
        + np.count_nonzero(dp & dg) / max(np.count_nonzero(dg), 1)
    )
    p = 0.25 * (
        np.count_nonzero(ap & ag) / max(np.count_nonzero(ap), 1)
        + np.count_nonzero(bp & bg) / max(np.count_nonzero(bp), 1)
        + np.count_nonzero(cp & cg) / max(np.count_nonzero(cp), 1)
        + np.count_nonzero(dp & dg) / max(np.count_nonzero(dp), 1)
    )
    if r + p == 0:
        return 0.0
    if return_p:
        return p
    if return_r:
        return r
    return 2 * (r * p) / (r + p)

def get_thresholds_and_weights(
    t_min: float, t_max: float, N: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate thresholds and weights for the given range.

    Args:
    ----
        t_min (float): Minimum threshold.
        t_max (float): Maximum threshold.
        N (int): Number of thresholds.

    Returns:
    -------
        Tuple[np.ndarray, np.ndarray]: Array of thresholds and corresponding weights.

    """
    thresholds = np.linspace(t_min, t_max, N)
    weights = thresholds / thresholds.sum()
    return thresholds, weights

def invert_depth(depth: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Inverts a depth map with numerical stability.

    Args:
    ----
        depth (np.ndarray): Depth map to be inverted.
        eps (float): Minimum value to avoid division by zero (default is 1e-6).

    Returns:
    -------
    np.ndarray: Inverted depth map.

    """
    inverse_depth = 1.0 / depth.clip(min=eps)
    return inverse_depth

def SI_boundary_F1(
    predicted_depth: np.ndarray,
    target_depth: np.ndarray,
    t_min: float = 1.05,
    t_max: float = 1.25,
    N: int = 10,
    mask: np.ndarray = None,
) -> float:
    """Calculate Scale-Invariant Boundary F1 Score for depth-based ground-truth.

    Args:
    ----
        predicted_depth (np.ndarray): Predicted depth matrix.
        target_depth (np.ndarray): Ground truth depth matrix.
        t_min (float, optional): Minimum threshold. Defaults to 1.05.
        t_max (float, optional): Maximum threshold. Defaults to 1.25.
        N (int, optional): Number of thresholds. Defaults to 10.

    Returns:
    -------
        float: Scale-Invariant Boundary F1 Score.

    """
    thresholds, weights = get_thresholds_and_weights(t_min, t_max, N)
    f1_scores = np.array(
        [
            boundary_f1(invert_depth(predicted_depth), invert_depth(target_depth), t, mask=mask)
            for t in thresholds
        ]
    )
    return np.sum(f1_scores * weights)

def eval_depth(pred, target, mask, step=None, eval_in_disparity_space=False, config=None, fsubaperture=521.4052, baseline=0.0002708787058842817):
    """Evaluate depth predictions against ground truth using various metrics.

    Args:
    ----        
        pred (torch.Tensor): Predicted depth map.
        target (torch.Tensor): Ground truth depth
        mask (torch.Tensor): Validity mask indicating which pixels to consider in evaluation.
        step (int, optional): Current evaluation step (for logging purposes). Defaults to None.
        eval_in_disparity_space (bool, optional): Whether to evaluate in disparity space. Defaults to False.
        config (dict, optional): Configuration dictionary that may contain additional parameters. Defaults to None.
        fsubaperture (float, optional): Subaperture focal length in pixels, used for disparity conversion if eval_in_disparity_space is True. Defaults to 521.4052 from DDFF12.
        baseline (float, optional): Baseline in meters, used for disparity conversion if eval_in_disparity_space is True. Defaults to 0.0002708787058842817 from DDFF12
    Returns: dict: Dictionary containing various evaluation metrics.
    """
    assert pred.shape == target.shape == mask.shape

    F1 = SI_boundary_F1(pred.cpu().numpy(), target.cpu().numpy(), mask=mask.cpu().numpy())

    pred = pred[mask]
    target = target[mask]

    if eval_in_disparity_space:
        if fsubaperture is None or baseline is None:
            raise ValueError("fsubaperture and baseline (this is pixel wise disparity space i.e. DDFF) must be provided for evaluation in disparity space.")
        pred = fsubaperture * baseline / pred
        target = fsubaperture * baseline / target

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d1_05 = torch.sum(thresh < 1.05).float() / len(thresh)
    d1_15 = torch.sum(thresh < 1.15).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    # metrics[0, 3] = np.ma.divide(np.abs(pred_ - target), target).sum() / numPixels

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    mse = torch.mean(torch.pow(diff, 2))

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))
    

    return {'d1_05': d1_05.item(), 'd1_15': d1_15.item(), 'd1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 
            'mse': mse.item(), 'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item(), 'silog':silog.item(), 'F1': F1}