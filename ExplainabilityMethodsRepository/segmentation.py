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
            'instance_id','i','j','lon','lat','dem','wd_in','label'
            + either 'rain' (JSON string per row) OR columns 'rain_0'..'rain_{T-1}'.
        T: optional int, number of timesteps. If None, T is inferred from columns:
            - if 'rain_0' present, T = max t+1 found,
            - else if 'rain' (json) column present, T will be inferred from first row's json length.
        C: number of channels (default 4).
        patch_size: optional (H,W). If provided, used for every instance; else inferred per-instance
            from the max i/j in each instance group (max + 1).
        fill_value: value to fill missing pixels (dem, wd_in, rain, label).
        infer_T_from_columns: if True and T is None, attempt to infer T automatically.

    Returns:
        instances: List[np.ndarray], each shape (T, C, H, W), dtype float32
        x_coords_list: List[np.ndarray], each shape (H, W) float32 (lon)
        y_coords_list: List[np.ndarray], each shape (H, W) float32 (lat)
        labels_list: List[np.ndarray], each shape (1, H, W) float32
    """
    # Validate required columns
    required = {"instance_id","i","j","lon","lat","dem","wd_in","label"}
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
            max_i = int(g['i'].max())
            max_j = int(g['j'].max())
            H = max_i + 1
            W = max_j + 1

        # allocate arrays
        arr = np.full((T, C, H, W), fill_value, dtype=np.float32)
        # default mask = 1 (no-data)
        mask_arr = np.ones((H, W), dtype=np.float32)
        # x/y coords default nan
        xcoords = np.full((H, W), np.nan, dtype=np.float32)
        ycoords = np.full((H, W), np.nan, dtype=np.float32)
        labels = np.full((H, W), fill_value, dtype=np.float32)

        # extract indices and values as numpy arrays for vectorized assignment
        i_idx = g['i'].to_numpy(dtype=np.int32)
        j_idx = g['j'].to_numpy(dtype=np.int32)

        # bounds check
        if np.any(i_idx < 0) or np.any(j_idx < 0):
            raise ValueError(f"Negative i/j indices in instance {inst_id}")
        if np.any(i_idx >= H) or np.any(j_idx >= W):
            # this can happen if patch_size was given and DataFrame implies larger. handle by resizing:
            # we'll re-allocate bigger arrays to fit all indices.
            H_new = max(H, int(i_idx.max())+1)
            W_new = max(W, int(j_idx.max())+1)
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
        lon_vals = g['lon'].to_numpy(dtype=np.float32)
        lat_vals = g['lat'].to_numpy(dtype=np.float32)
        label_vals = g['label'].to_numpy(dtype=np.float32)

        arr[0, 0, i_idx, j_idx] = dem_vals          # DEM at time 0 (stored static)
        arr[0, 2, i_idx, j_idx] = wd_in_vals        # wd_in channel
        xcoords[i_idx, j_idx] = lon_vals
        ycoords[i_idx, j_idx] = lat_vals
        labels[i_idx, j_idx] = label_vals
        mask_arr[i_idx, j_idx] = 0.0                # mark valid pixels

        # rain handling
        if rain_cols:
            # expect rain_0...rain_{T-1}
            for t in range(T):
                col = f"rain_{t}"
                if col not in g.columns:
                    raise ValueError(f"Missing expected rain column '{col}' in DataFrame.")
                vals = g[col].to_numpy(dtype=np.float32)
                arr[t, 3, i_idx, j_idx] = vals
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
                arr[t, 3, i_idx, j_idx] = parsed_arr[:, t]
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
