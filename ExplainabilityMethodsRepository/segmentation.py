from typing import Dict, Tuple, List, Optional, Any
import json

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import logging
logger = logging.getLogger(__name__)

def treat_input(input_instance, device, req_grad=False):
    input_instance_treated = torch.tensor(input_instance, requires_grad=req_grad).to(device).unsqueeze(0)
    return input_instance_treated

def treat_mask(mask, device, req_grad=False):
    mask_treated = torch.tensor(mask, requires_grad=req_grad).to(device)
    return mask_treated

def model_wrapper(input_tensor: torch.Tensor,
                  mask: torch.Tensor,
                  model: torch.nn.Module) -> torch.Tensor:
    """
    Wrap model to produce a scalar from shape:
      input_tensor: (1, T, C, H, W)
      mask:         (1, 1, H, W) where 1 = no-data, 0 = valid
      model outputs (1, 1, H, W)
    The scalar is the mean output over all non-masked pixels.
    """
    # Forward pass
    out = model(input_tensor)               # (1, 1, 512, 512)

    # Mask: shape (1, 1, 512, 512). Same as output, so we can elementwise multiply.
    valid_mask = (mask == 0)              # True where valid pixels are

    # Sum and count non-masked pixels
    logger.info(f"Output shape: {out.shape}, valid_mask shape: {valid_mask.shape}")
    sum_valid = (out * valid_mask).sum(dim=(2, 3))           # shape (1,1)
    count_valid = valid_mask.sum(dim=(2, 3))    # shape (1,1)
    count_valid = count_valid.clamp(min=1)  # avoid div by 0

    # Mean over valid pixels
    avg = sum_valid / count_valid     # shape (1,1)

    return avg.squeeze(1)     # -> (1,)

def model_wrapper_roi(
    input_tensor: torch.Tensor,  # shape: (B, T, C, H, W)
    mask: torch.Tensor,          # shape: (B, 1, H, W), 1 = no-data, 0 = valid
    region_mask: torch.Tensor,   # shape: (B, 1, H, W), 1 = in-ROI, 0 = out
    model: torch.nn.Module
) -> torch.Tensor:             # returns shape: (B,)
    """
    Runs `model` on input_tensor and returns the mean of outputs over pixels that
    are both valid (mask == 0) and inside the ROI (region_mask == 1).
    """
    out = model(input_tensor)           # (B, 1, H, W)
    assert out.ndim == 4 and mask.ndim == 4 and region_mask.ndim == 4

    # Boolean masks: shape (B, 1, H, W)
    valid = mask == 0                   # True where we have data
    in_roi = region_mask == 1           # True where we want to explain
    select = valid & in_roi             # True only where both hold

    # Sum and count valid ROI pixels, per sample
    summed = (out * select).sum(dim=(2, 3))        # (B, 1)
    counts = select.sum(dim=(2, 3)).clamp(min=1)    # (B, 1)

    means = summed / counts                        # (B, 1)
    return means.squeeze(1)                        # → (B,)

@torch.no_grad()
def compute_csi_rmse(
    model: torch.nn.Module,
    inputs: torch.Tensor,    # (N, T, C, H, W)
    targets: torch.Tensor,   # (N, 1, H, W)
    masks: torch.Tensor,     # (N, 1, H, W), 1=no-data, 0=valid
    flooded_min: float = 0.01,
    flooded_max: float = float('inf'),
    device: torch.device = None
) -> Dict[str, float]:
    """
    Returns a dict {'rmse': float, 'csi': float}, where CSI is computed
    by counting as 'flooded' any pixel y in [flooded_min, flooded_max).
    """
    # Move model + data to device
    dev = device or inputs.device
    model   = model.to(dev).eval()
    inputs  = inputs.to(dev)
    targets = targets.to(dev)
    masks   = masks.to(dev)

    # Forward pass
    preds = model(inputs).squeeze(1)   # → (N, H, W)
    truths = targets.squeeze(1)        # → (N, H, W)
    valid = (masks.squeeze(1) == 0)    # True where data is valid

    # 1) RMSE over valid pixels
    y_true_flat = truths[valid].cpu().numpy().ravel()
    y_pred_flat = preds[valid].cpu().numpy().ravel()
    rmse = float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)))

    # 2) CSI with [flooded_min, flooded_max)
    gt_flood = np.logical_and(y_true_flat  >= flooded_min,
                              y_true_flat  <  flooded_max)
    pr_flood = np.logical_and(y_pred_flat  >= flooded_min,
                              y_pred_flat  <  flooded_max)

    tp = np.logical_and(gt_flood, pr_flood).sum()
    fp = np.logical_and(~gt_flood, pr_flood).sum()
    fn = np.logical_and(gt_flood, ~pr_flood).sum()

    denom = tp + fp + fn
    csi = (tp / denom) if denom > 0 else 0.0

    return {'rmse': rmse, 'csi': csi}

@torch.no_grad()
def replacement_feature_importance(
    model: torch.nn.Module,
    inputs: torch.Tensor,     # (N, T, C, H, W)
    targets: torch.Tensor,    # (N, 1, H, W)
    masks: torch.Tensor,      # (N, 1, H, W)
    flooded_min: float = 0.01,
    flooded_max: float = float('inf'),
    n_trials: int = 10,
    device: torch.device = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perturb each channel by shuffling its T x H x W values per sample, over n_trials.
    Returns:
      - importance_rmse_mean: mean ΔRMSE per channel (shape C,)
      - importance_rmse_std:  std  ΔRMSE per channel (shape C,)
      - importance_csi_mean: mean ΔCSI  per channel (shape C,)
      - importance_csi_std:  std  ΔCSI  per channel (shape C,)
    """
    # Move to device
    dev = device or inputs.device
    inputs  = inputs.to(dev)
    targets = targets.to(dev)
    masks   = masks.to(dev)
    model   = model.to(dev).eval()

    # Baseline metrics
    baseline = compute_csi_rmse(
        model, inputs, targets, masks,
        flooded_min=flooded_min,
        flooded_max=flooded_max,
        device=dev
    )
    baseline_rmse = baseline['rmse']
    baseline_csi  = baseline['csi']

    N, T, C, H, W = inputs.shape

    # Storage for trial results
    rmse_trials = np.zeros((C, n_trials), dtype=float)
    csi_trials  = np.zeros((C, n_trials), dtype=float)

    # Perturbations
    for c in range(C):
        for t in range(n_trials):
            inp_mod = inputs.clone()
            flat_len = T * H * W
            # shuffle channel c per sample
            for i in range(N):
                feat = inp_mod[i, :, c].reshape(flat_len)
                perm = torch.randperm(flat_len, device=dev)
                inp_mod[i, :, c] = feat[perm].reshape(T, H, W)
            # compute metrics
            m = compute_csi_rmse(
                model, inp_mod, targets, masks,
                flooded_min=flooded_min,
                flooded_max=flooded_max,
                device=dev
            )
            rmse_trials[c, t] = m['rmse']
            csi_trials[c, t]  = m['csi']

    # Δ metrics
    delta_rmse = rmse_trials - baseline_rmse     # shape (C, n_trials)
    delta_csi  = baseline_csi - csi_trials       # shape (C, n_trials)

    # Mean and std over trials
    importance_rmse_mean = delta_rmse.mean(axis=1)
    importance_rmse_std  = delta_rmse.std(axis=1, ddof=1)
    importance_csi_mean  = delta_csi.mean(axis=1)
    importance_csi_std   = delta_csi.std(axis=1, ddof=1)

    return (
        importance_rmse_mean,
        importance_rmse_std,
        importance_csi_mean,
        importance_csi_std
    )

# ------------------------------
# Batched metric computation
# ------------------------------
@torch.no_grad()
def compute_csi_rmse_batched(
    model: torch.nn.Module,
    inputs: torch.Tensor,   # (N, T, C, H, W)
    targets: torch.Tensor,  # (N, 1, H, W) or (N, H, W)
    masks: torch.Tensor,    # (N, 1, H, W) or (N, H, W)  where 1 = nodata, 0 = valid
    flooded_min: float = 0.01,
    flooded_max: float = 5.,
    batch_size: int = 4,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Compute RMSE and CSI over potentially large inputs by iterating in batches.
    Returns {'rmse': float, 'csi': float}.
    """
    dev = device or (next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else torch.device('cpu'))
    model = model.to(dev).eval()

    N = inputs.shape[0]
    total_sse = 0.0
    total_valid = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for i in range(0, N, batch_size):
        b_inputs = inputs[i: i + batch_size].to(dev)
        b_targets = targets[i: i + batch_size].to(dev)
        b_masks = masks[i: i + batch_size].to(dev)

        # forward
        preds = model(b_inputs)   # expect (b,1,H,W)
        if preds.ndim == 4 and preds.shape[1] == 1:
            preds = preds.squeeze(1)  # (b, H, W)
        else:
            preds = preds.squeeze(1)

        truths = b_targets
        if truths.ndim == 4 and truths.shape[1] == 1:
            truths = truths.squeeze(1)

        valid = (b_masks.squeeze(1) == 0) if b_masks.ndim == 4 else (b_masks == 0)
        valid = valid.bool()

        # SSE and counts (use torch, then convert)
        diff = (preds - truths) * valid.float()
        sse = (diff ** 2).sum().item()
        n_valid = int(valid.sum().item())

        total_sse += sse
        total_valid += n_valid

        # CSI components
        gt_f = (truths >= flooded_min) & (truths < flooded_max)
        pr_f = (preds >= flooded_min) & (preds < flooded_max)

        # mask out invalid
        gt_f = gt_f & valid
        pr_f = pr_f & valid

        tp = int((gt_f & pr_f).sum().item())
        fp = int((~gt_f & pr_f).sum().item())
        fn = int((gt_f & ~pr_f).sum().item())

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # finalize
    rmse = float(np.sqrt(total_sse / total_valid)) if total_valid > 0 else 0.0
    denom = total_tp + total_fp + total_fn
    csi = float(total_tp / denom) if denom > 0 else 0.0

    return {'rmse': rmse, 'csi': csi}

# ------------------------------
# Batched replacement/permutation FI
# ------------------------------
@torch.no_grad()
def replacement_feature_importance_batched(
    model: torch.nn.Module,
    inputs: torch.Tensor,     # (N, T, C, H, W)
    targets: torch.Tensor,    # (N, 1, H, W)
    masks: torch.Tensor,      # (N, 1, H, W)
    flooded_min: float = 0.01,
    flooded_max: float = float('inf'),
    n_trials: int = 10,
    batch_size: int = 4,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batched replacement/permutation feature importance.

    Returns (delta_rmse, delta_csi), each shape (C, n_trials), where:
      delta_rmse[c,t] = rmse_perturbed - baseline_rmse
      delta_csi[c,t]  = baseline_csi - csi_perturbed
    (Matches previous function's sign conventions.)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    dev = device or (next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else torch.device('cpu'))
    model = model.to(dev).eval()

    # Ensure shapes
    if inputs.ndim != 5:
        raise ValueError("inputs must be (N, T, C, H, W)")
    N, T, C, H, W = inputs.shape

    # Compute baseline metrics (batched)
    baseline = compute_csi_rmse_batched(
        model, inputs, targets, masks,
        flooded_min=flooded_min, flooded_max=flooded_max,
        batch_size=batch_size, device=dev
    )
    baseline_rmse = baseline['rmse']
    baseline_csi = baseline['csi']

    # prepare storage
    rmse_trials = np.zeros((C, n_trials), dtype=float)
    csi_trials  = np.zeros((C, n_trials), dtype=float)

    # We'll operate batch-by-batch for each trial and channel
    flat_len = T * H * W

    # Iterate channels and trials
    for c in range(C):
        for t in range(n_trials):
            # We'll accumulate sse/counts and tp/fp/fn across batches for this perturbed dataset
            total_sse = 0.0
            total_valid = 0
            total_tp = 0
            total_fp = 0
            total_fn = 0

            # Iterate dataset in batches and perturb only the current batch in memory
            for i in range(0, N, batch_size):
                b_inputs = inputs[i: i + batch_size].clone().to(dev)   # clone so we can permute in-place
                b_targets = targets[i: i + batch_size].to(dev)
                b_masks = masks[i: i + batch_size].to(dev)

                b_size = b_inputs.shape[0]

                # For each sample in the batch, permute the chosen channel across flattened space
                # (this reproduces your prior behaviour: shuffle the T*H*W entries for channel c).
                # This loop is per-sample but operates only on a small batch in memory.
                for bi in range(b_size):
                    feat = b_inputs[bi, :, c].reshape(-1)           # length flat_len
                    perm = torch.randperm(flat_len, device=dev)
                    b_inputs[bi, :, c] = feat[perm].reshape(T, H, W)

                # forward on perturbed batch
                preds = model(b_inputs)
                if preds.ndim == 4 and preds.shape[1] == 1:
                    preds = preds.squeeze(1)

                truths = b_targets
                if truths.ndim == 4 and truths.shape[1] == 1:
                    truths = truths.squeeze(1)

                valid = (b_masks.squeeze(1) == 0) if b_masks.ndim == 4 else (b_masks == 0)
                valid = valid.bool()

                # accumulate sse & counts
                diff = (preds - truths) * valid.float()
                sse = (diff ** 2).sum().item()
                n_valid = int(valid.sum().item())

                total_sse += sse
                total_valid += n_valid

                # accumulate CSI counts
                gt_f = (truths >= flooded_min) & (truths < flooded_max)
                pr_f = (preds >= flooded_min) & (preds < flooded_max)
                gt_f = gt_f & valid
                pr_f = pr_f & valid

                total_tp += int((gt_f & pr_f).sum().item())
                total_fp += int((~gt_f & pr_f).sum().item())
                total_fn += int((gt_f & ~pr_f).sum().item())

            # finalize this trial/channel
            rmse_pert = float(np.sqrt(total_sse / total_valid)) if total_valid > 0 else 0.0
            denom = total_tp + total_fp + total_fn
            csi_pert = float(total_tp / denom) if denom > 0 else 0.0

            rmse_trials[c, t] = rmse_pert
            csi_trials[c, t]  = csi_pert

    # compute deltas (keep same sign convention as before)
    delta_rmse = rmse_trials - baseline_rmse
    delta_csi = baseline_csi - csi_trials

    return delta_rmse, delta_csi

def parse_instance_from_request(request) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    query = request.query
    query = pd.read_csv(query)
    parsed_query = df_to_instances(query)
    for q in parsed_query:
        if len(q) != 1:
            raise ValueError("More than one instance found in request.")
    instance, x_coords, y_coords, label = parsed_query[0][0], parsed_query[1][0], parsed_query[2][0], parsed_query[3][0]
    mask = (instance[[0],1,:,:] == 1).astype(np.float32)  # 1 = no-data, 0 = valid
    return instance, x_coords, y_coords, label, mask

def df_to_instances(
    df: pd.DataFrame,
    T: Optional[int] = None,
    C: int = 4,
    patch_size: Optional[Tuple[int,int]] = None,
    fill_value: float = 0.0,
    infer_T_from_columns: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Reconstruct instances, x_coords_list, y_coords_list, labels_list from DataFrame `df`
    produced by `instances_to_df`.

    Args:
        df: pandas DataFrame with rows for valid pixels. Required cols:
            'instance_id','row','col','longitude','latitude','dem','wd_in','label'
            + either 'rain' (JSON string per row) OR columns 'rain_0'..'rain_{T-1}'.
        T: optional int, number of timesteps. If None, T is inferred from columns:
            - if 'rain_0' present, T = max t+1 found,
            - else if 'rain' (json) column present, T will be inferred from first row's json length.
        C: number of channels (default 4).
        patch_size: optional (H,W). If provided, used for every instance; else inferred per-instance
            from the max row/col in each instance group (max + 1).
        fill_value: value to fill missing pixels (dem, wd_in, rain, label).
        infer_T_from_columns: if True and T is None, attempt to infer T automatically.

    Returns:
        instances: List[np.ndarray], each shape (T, C, H, W), dtype float32
        x_coords_list: List[np.ndarray], each shape (H, W) float32 (longitude)
        y_coords_list: List[np.ndarray], each shape (H, W) float32 (latitude)
        labels_list: List[np.ndarray], each shape (1, H, W) float32
    """
    # Validate required columns
    required = {"instance_id","row","col","longitude","latitude","dem","wd_in","label"}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Detect rain format and infer T if needed
    rain_json_mode = 'rain' in df.columns and not any(col.startswith('rain_') for col in df.columns)
    rain_cols = [col for col in df.columns if col.startswith('rain_')]
    if T is None and infer_T_from_columns:
        if rain_cols:
            # find max t
            ts = []
            for c in rain_cols:
                try:
                    t = int(c.split('_',1)[1])
                    ts.append(t)
                except Exception:
                    pass
            if not ts:
                raise ValueError("Found rain_ columns but couldn't parse timesteps.")
            T = max(ts) + 1
        elif rain_json_mode:
            # infer from first non-null row
            sample = df['rain'].dropna().iloc[0]
            try:
                parsed = json.loads(sample) if isinstance(sample, str) else list(sample)
                T = len(parsed)
            except Exception:
                raise ValueError("Couldn't infer T from 'rain' JSON column.")
        else:
            raise ValueError("T not provided and no rain information to infer from.")
    if T is None:
        raise ValueError("T must be provided or inferable from DataFrame (rain columns).")

    instances = []
    x_coords_list = []
    y_coords_list = []
    labels_list = []

    # iterate per-instance
    grouped = df.groupby('instance_id', sort=True)
    for inst_id, g in grouped:
        # compute (H,W)
        if patch_size is not None:
            H, W = patch_size
        else:
            max_row = int(g['row'].max())
            max_col = int(g['col'].max())
            H = max_row + 1
            W = max_col + 1

        # allocate arrays
        arr = np.full((T, C, H, W), fill_value, dtype=np.float32)
        # default mask = 1 (no-data)
        mask_arr = np.ones((H, W), dtype=np.float32)
        # x/y coords default nan
        xcoords = np.full((H, W), np.nan, dtype=np.float32)
        ycoords = np.full((H, W), np.nan, dtype=np.float32)
        labels = np.full((H, W), fill_value, dtype=np.float32)

        # extract indices and values as numpy arrays for vectorized assignment
        row_idx = g['row'].to_numpy(dtype=np.int32)
        col_idx = g['col'].to_numpy(dtype=np.int32)

        # bounds check
        if np.any(row_idx < 0) or np.any(col_idx < 0):
            raise ValueError(f"Negative row/col indices in instance {inst_id}")
        if np.any(row_idx >= H) or np.any(col_idx >= W):
            # this can happen if patch_size was given and DataFrame implies larger. handle by resizing:
            # we'll re-allocate bigger arrays to fit all indices.
            H_new = max(H, int(row_idx.max())+1)
            W_new = max(W, int(col_idx.max())+1)
            arr_new = np.full((T, C, H_new, W_new), fill_value, dtype=np.float32)
            arr_new[:, :, :H, :W] = arr
            arr = arr_new
            mask_new = np.ones((H_new, W_new), dtype=np.float32)
            mask_new[:H, :W] = mask_arr
            mask_arr = mask_new
            x_new = np.full((H_new, W_new), np.nan, dtype=np.float32)
            x_new[:H, :W] = xcoords
            xcoords = x_new
            y_new = np.full((H_new, W_new), np.nan, dtype=np.float32)
            y_new[:H, :W] = ycoords
            ycoords = y_new
            labels_new = np.full((H_new, W_new), fill_value, dtype=np.float32)
            labels_new[:H, :W] = labels
            labels = labels_new
            H, W = H_new, W_new

        # fill per-pixel scalar columns
        dem_vals = g['dem'].to_numpy(dtype=np.float32)
        wd_in_vals = g['wd_in'].to_numpy(dtype=np.float32)
        lon_vals = g['longitude'].to_numpy(dtype=np.float32)
        lat_vals = g['latitude'].to_numpy(dtype=np.float32)
        label_vals = g['label'].to_numpy(dtype=np.float32)

        arr[0, 0, row_idx, col_idx] = dem_vals          # DEM at time 0 (stored static)
        arr[0, 2, row_idx, col_idx] = wd_in_vals        # wd_in channel
        xcoords[row_idx, col_idx] = lon_vals
        ycoords[row_idx, col_idx] = lat_vals
        labels[row_idx, col_idx] = label_vals
        mask_arr[row_idx, col_idx] = 0.0                # mark valid pixels

        # rain handling
        if rain_cols:
            # expect rain_0...rain_{T-1}
            for t in range(T):
                col = f"rain_{t}"
                if col not in g.columns:
                    raise ValueError(f"Missing expected rain column '{col}' in DataFrame.")
                vals = g[col].to_numpy(dtype=np.float32)
                arr[t, 3, row_idx, col_idx] = vals
        elif rain_json_mode:
            # parse each JSON string into length-T list and assign
            rain_series = g['rain'].to_numpy()
            # parse all
            parsed = []
            for s in rain_series:
                if isinstance(s, str):
                    row = json.loads(s)
                else:
                    # already list-like
                    row = list(s)
                if len(row) != T:
                    raise ValueError(f"Rain length mismatch for instance {inst_id}: expected {T}, got {len(row)}")
                parsed.append(row)
            parsed_arr = np.array(parsed, dtype=np.float32)  # shape (n_valid, T)
            # assign per time
            for t in range(T):
                arr[t, 3, row_idx, col_idx] = parsed_arr[:, t]
        else:
            # no rain info present; leave as fill_value
            pass

        # set mask channel in arr (channel 1)
        arr[0, 1, :, :] = mask_arr.astype(np.float32)

        # ensure labels shape is (1,H,W)
        labels_out = labels.reshape(1, H, W).astype(np.float32)

        instances.append(arr)
        x_coords_list.append(xcoords)
        y_coords_list.append(ycoords)
        labels_list.append(labels_out)

    return instances, x_coords_list, y_coords_list, labels_list

def compress_attributions(
    x_coords: np.ndarray,        # (H, W) lon or x
    y_coords: np.ndarray,        # (H, W) lat or y
    attribution_map: np.ndarray, # (H, W) float
    mask: np.ndarray,            # (1,H,W) or (H,W) with 1 = nodata, 0 = valid
    max_points: int = 20000,     # target max number of points to return
    roi_mask: Optional[np.ndarray] = None,  # (H,W) boolean, keep these always
    expand_radius: int = 0,      # 0 = no expansion, 1 = include 8-neighbors, etc.
    do_quantize: bool = True,    # quantize attributions to int16 + scale
    preserve_all_if_small: bool = True, # if valid pixels <= max_points return all
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Return compressed lists (x_vals, y_vals, z_vals) and metadata.

    z_vals are int16 if do_quantize True (strings of ints returned), plus meta 'scale' for reconstruction:
       real_z = z_int / 32767 * scale

    Returns:
      x_out, y_out, z_out, meta
    where x_out,y_out,z_out are numpy 1D arrays (dtype depends on quantize),
    and meta contains {'method':..., 'scale':..., 'selected_count':..., 'orig_valid_count':...}
    """
    # Normalize shapes
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask2d = mask[0]
    else:
        mask2d = mask.copy()
    H, W = attribution_map.shape
    assert x_coords.shape == (H, W) and y_coords.shape == (H, W), "coord shapes must match attribution_map"

    valid_mask = (mask2d == 0)
    valid_idx_flat = np.nonzero(valid_mask.ravel())[0]
    orig_valid_count = valid_idx_flat.size

    meta: Dict[str,Any] = {"orig_valid_count": int(orig_valid_count)}

    if orig_valid_count == 0:
        # nothing to return
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float), {"method":"none","orig_valid_count":0}

    # Flatten arrays for easy indexing
    flat_attr = attribution_map.ravel()
    flat_x = x_coords.ravel()
    flat_y = y_coords.ravel()

    # Quick path: if small enough, return all valid pixels
    if preserve_all_if_small and orig_valid_count <= max_points:
        sel = valid_idx_flat
        method = "all_valid"
    else:
        # Select top-K by absolute attribution
        K = min(max_points, orig_valid_count)
        abs_vals = np.abs(flat_attr[valid_idx_flat])

        if K >= orig_valid_count:
            top_idx_in_valid = np.arange(orig_valid_count)
        else:
            # argpartition to get indices of top-K (faster than full sort)
            # note: we want the K largest absolute values
            kth = orig_valid_count - K
            # find indices in valid_idx_flat that correspond to top-K
            part = np.argpartition(abs_vals, -K)[-K:]
            top_idx_in_valid = part

        sel = valid_idx_flat[top_idx_in_valid]
        method = "topk_abs"

        # optionally union with roi_mask
        if roi_mask is not None:
            # ensure roi_mask boolean and same shape
            roi_bool = (roi_mask != 0)
            roi_idx = np.nonzero(roi_bool.ravel())[0]
            # union
            if roi_idx.size:
                sel = np.unique(np.concatenate([sel, roi_idx]))

        # optionally expand neighborhood by simple square dilation
        if expand_radius and sel.size:
            # convert flat indices to coordinates
            ii, jj = np.unravel_index(sel, (H, W))
            # bucket boolean for selected
            sel_bool = np.zeros(H*W, dtype=bool)
            sel_bool[sel] = True
            # perform dilation by radius r using simple iteration (radius small)
            r = int(expand_radius)
            sel_bool2 = sel_bool.copy()
            for di in range(-r, r+1):
                for dj in range(-r, r+1):
                    if di == 0 and dj == 0:
                        continue
                    # shift indices
                    ii2 = np.clip(ii + di, 0, H-1)
                    jj2 = np.clip(jj + dj, 0, W-1)
                    sel_bool2[np.ravel_multi_index((ii2, jj2), (H, W))] = True
            sel = np.nonzero(sel_bool2)[0]

        # finally ensure sel contains only originally valid pixels
        sel = np.intersect1d(sel, valid_idx_flat, assume_unique=True)
        # if still too big (shouldn't), reduce by taking top-K of current selection
        if sel.size > max_points:
            sel_abs = np.abs(flat_attr[sel])
            sel_part = np.argpartition(sel_abs, -max_points)[-max_points:]
            sel = sel[sel_part]

    # Prepare outputs
    x_sel = flat_x[sel]
    y_sel = flat_y[sel]
    z_sel = flat_attr[sel]

    # Quantize z to int16 if requested
    if do_quantize and z_sel.size > 0:
        scale = float(np.max(np.abs(z_sel)))
        if scale == 0.0:
            z_int = np.zeros(z_sel.shape, dtype=np.int16)
            meta.update({"scale": 0.0, "quantized": True, "method": method})
        else:
            # scale to int16 range [-32767, 32767]
            scaled = z_sel / scale * 32767.0
            # rounding and clipping
            z_int = np.rint(scaled).astype(np.int32)
            z_int = np.clip(z_int, -32767, 32767).astype(np.int16)
            meta.update({"scale": float(scale), "quantized": True, "method": method})
        x_out = x_sel
        y_out = y_sel
        z_out = z_int
    else:
        # keep floats
        x_out = x_sel
        y_out = y_sel
        z_out = z_sel.astype(np.float32)
        meta.update({"scale": None, "quantized": False, "method": method})

    meta["selected_count"] = int(x_out.size)
    meta["orig_valid_count"] = int(orig_valid_count)
    return x_out, y_out, z_out, meta

def attributions_to_filtered_long_df(
    attributions: np.ndarray,             # (T, C, H, W) or (1, T, C, H, W)
    x_coords: np.ndarray,                 # (H, W)
    y_coords: np.ndarray,                 # (H, W)
    mask: Optional[np.ndarray] = None,    # (H,W) or (1,H,W), 1 = nodata, 0 = valid
    channel_names: Optional[List[str]] = None,
    max_rows: int = 20000,
    roi_mask: Optional[np.ndarray] = None,      # (H,W) boolean, keep these pixels (all times)
    preserve_all_if_small: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convert full attributions (T,C,H,W) to a long DataFrame filtered by top-K importance.

    Returns (df, meta) where df columns = ['x','y','time', <channel_names...>].

    Selection metric (per time,pixel) = sum(|attr| across channels).
    """
    # -- normalize attributions --
    attr = np.asarray(attributions)
    if attr.ndim == 5 and attr.shape[0] == 1:
        attr = attr[0]   # drop batch dim
    if attr.ndim != 4:
        raise ValueError(f"attributions must be shape (T,C,H,W) or (1,T,C,H,W), got {attr.shape}")

    T, C, H, W = attr.shape

    # coords validation
    if x_coords.shape != (H, W) or y_coords.shape != (H, W):
        raise ValueError("x_coords and y_coords must have shape (H, W) matching attributions")

    # channel names
    if channel_names is None:
        channel_names = [f"ch_{c}" for c in range(C)]
    else:
        if len(channel_names) != C:
            raise ValueError("channel_names length must equal number of channels C")

    # normalize mask
    if mask is not None:
        m = np.asarray(mask)
        if m.ndim == 3 and m.shape[0] == 1:
            m = m[0]
        if m.shape != (H, W):
            raise ValueError("mask must have shape (H, W) or (1, H, W)")
        valid_pixel_flat = (m.ravel() == 0)   # length H*W
    else:
        valid_pixel_flat = np.ones(H*W, dtype=bool)

    # normalize roi_mask
    if roi_mask is not None:
        r = np.asarray(roi_mask)
        if r.ndim == 3 and r.shape[0] == 1:
            r = r[0]
        if r.shape != (H, W):
            raise ValueError("roi_mask must have shape (H, W) or (1, H, W)")
        roi_pixel_flat = (r.ravel() != 0)
    else:
        roi_pixel_flat = np.zeros(H*W, dtype=bool)

    # Build flattened coordinate/time arrays (length L = T*H*W)
    L = T * H * W
    flat_x = np.tile(x_coords.ravel(), T).astype(np.float32)   # (L,)
    flat_y = np.tile(y_coords.ravel(), T).astype(np.float32)
    flat_time = np.repeat(np.arange(T, dtype=np.int32), H*W)   # (L,)

    # Build per-channel flat arrays (each shape L,)
    channel_flat = {}
    for c in range(C):
        channel_flat[channel_names[c]] = attr[:, c, :, :].reshape(-1).astype(np.float32)

    # Build valid selector over time-pixel entries
    valid_time_flat = np.tile(valid_pixel_flat, T)  # (L,)
    valid_idx = np.nonzero(valid_time_flat)[0]
    orig_valid_count = valid_idx.size

    meta: Dict[str, Any] = {"orig_valid_count": int(orig_valid_count)}

    if orig_valid_count == 0:
        # nothing to return: empty df
        cols = ["x", "y", "time"] + channel_names
        return pd.DataFrame(columns=cols), meta

    # Compute importance metric per time-pixel: sum absolute across channels
    # shape (T,H,W) -> flatten to (L,)
    importance_metric = np.sum(np.abs(attr), axis=1).reshape(-1)    # sum over channel axis -> (T,H,W) then flatten

    # Quick path: return everything if small
    if preserve_all_if_small and orig_valid_count <= max_rows:
        sel_idx = valid_idx
        method = "all_valid"
    else:
        # pick top-K among valid by importance_metric
        K = min(max_rows, orig_valid_count)
        abs_vals = importance_metric[valid_idx]
        if K >= orig_valid_count:
            top_in_valid = np.arange(orig_valid_count)
        else:
            top_in_valid = np.argpartition(abs_vals, -K)[-K:]
        sel_idx = valid_idx[top_in_valid]
        method = "topk_sumabs"

        # union with roi_mask: include all times for ROI pixels
        if roi_pixel_flat.any():
            roi_time_idx = np.nonzero(np.tile(roi_pixel_flat, T))[0]
            if roi_time_idx.size:
                sel_idx = np.unique(np.concatenate([sel_idx, roi_time_idx]))

        # ensure only valid entries (defensive)
        sel_idx = np.intersect1d(sel_idx, valid_idx, assume_unique=True)

        # if still too large, reduce to top-K by importance among sel_idx
        if sel_idx.size > max_rows:
            sel_abs = importance_metric[sel_idx]
            keep = np.argpartition(sel_abs, -max_rows)[-max_rows:]
            sel_idx = sel_idx[keep]

    # Sort selected indices by descending importance for stable ordering (optional)
    order = np.argsort(-importance_metric[sel_idx])
    sel_idx = sel_idx[order]

    # Build DataFrame from selected rows
    data = {
        "x": flat_x[sel_idx],
        "y": flat_y[sel_idx],
        "time": flat_time[sel_idx].astype(np.int32),
    }
    for cname in channel_names:
        data[cname] = channel_flat[cname][sel_idx]

    df = pd.DataFrame(data)

    meta.update({
        "method": method,
        "selected_count": int(df.shape[0]),
        "max_rows": int(max_rows)
    })

    return df, meta
