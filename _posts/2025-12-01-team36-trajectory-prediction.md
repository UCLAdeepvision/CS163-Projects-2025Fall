---
layout: post
comments: false
title: Multimodal Trajectory Prediction
author: Albert Le
date: 2025-12-01
---

<div style="flex: 1 1 45%; min-width: 300px; text-align: center;">
    <img src="{{ '/assets/images/team36/scene_1679_multimodal.gif' | relative_url }}" alt="Scene 1679 Multimodal Prediction" style="width: 100%; border: 1px solid #ccc; border-radius: 5px;">
    <p><em>My 2 models predictions on scene 1679 from L5Kit</em></p>
</div>

> **Abstract**  
> Predicting the future trajectories of surrounding agents is a fundamental challenge in autonomous driving systems. In this project, I explore deep learning-based trajectory prediction using the Lyft Level-5 Motion Prediction dataset. I develop 2 models, a ResNet-based raster trajectory regressor, and a **multimodal raster + agent-history fusion model** capable of producing six future trajectory hypotheses with associated confidence scores. The final model achieves significant performance improvements in ADE, FDE, and Miss Rate metrics while producing interpretable, multimodal predictions essential for safe autonomous navigation.

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

## Introduction

<div style="flex: 1 1 45%; min-width: 300px; text-align: center;">
    <img src="{{ '/assets/images/team36/scene_12138_multimodal.gif' | relative_url }}" alt="Scene 12138 Multimodal Prediction" style="width: 100%; border: 1px solid #ccc; border-radius: 5px;">
    <p><em>Scene 12138 from L5Kit: Comparison in a straight line.</em></p>
  </div>

### The Challenge of Motion Prediction

Predicting the future motion of surrounding agents (vehicles, pedestrians, and cyclists) is one of the most critical challenges in autonomous driving. Unlike traditional robotics problems with deterministic outcomes, motion prediction in driving scenarios is inherently **multimodal**: a vehicle approaching an intersection might turn left, turn right, go straight, or stop, whereas a pedestrian can do those and also something completely unpredictable like jumping on your car. A robust prediction system must capture this uncertainty rather than committing to a single deterministic future.

The key challenges in trajectory prediction include:

1. **Multimodality**: Agents can take multiple plausible paths
2. **Scene Context**: Road geometry, lane markings, and traffic rules constrain possible futures
3. **Agent Interactions**: Surrounding vehicles influence each other's behavior

### My Approach

I start with building models of increasing complexity to address these challenges:

1. **Kinematic Baselines**
2. **MLP Baseline**
3. **ResNet Single-Trajectory Model**
4. **Multimodal Raster + History Fusion Model**

---
## Dataset & Problem Formulation

### The Lyft Level-5 Motion Prediction Dataset

I will use the **Lyft Level-5 Motion Prediction Dataset**, one of the largest publicly available datasets for autonomous vehicle motion prediction. This dataset includes scenes (driving episodes), frames (ego poses over time), tracked agents with positions, orientations, bounding boxes, label probabilities, and traffic light faces, all stored in `.zarr` format, along with BEV-ready map assets (`aerial_map`, `semantic_map`), train/validate/test splits (`train.zarr`, `validate.zarr`, `test.csv`, `mask.npz`), and an optional larger training set (`train_full.csv`).[^1]

[^1]: https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/data

### Mathematical Problem Formulation

Given:
- Agent history positions: $$\mathbf{H} = \{(x_t, y_t)\}_{t=-T_h}^{0}$$ where $$T_h = 10$$ frames
- Rasterized scene context: $$\mathbf{I} \in \mathbb{R}^{C \times H \times W}$$ (semantic/satellite map)
- Agent availability mask: $$\mathbf{a}^h \in \{0,1\}^{T_h}$$

Predict:
- Future trajectory: $$\hat{\mathbf{Y}} = \{(\hat{x}_t, \hat{y}_t)\}_{t=1}^{T_f}$$ where $$T_f = 50$$ frames

For multimodal prediction, $$M$$ trajectory hypotheses with associated confidences can be calculated with:
- Trajectories: $$\{\hat{\mathbf{Y}}^{(m)}\}_{m=1}^{M}$$ where $$M = 6$$
- Confidences: $$\{c^{(m)}\}_{m=1}^{M}$$ where $$\sum_m c^{(m)} = 1$$

### Evaluation Metrics

All models are evaluated using three standard metrics implemented in the evaluation module:

#### **Average Displacement Error (ADE)**  
ADE measures the mean L2 distance between predicted and ground-truth positions over all valid future timesteps:

$$ \text{ADE} = \frac{1}{T_f} \sum_{t=1}^{T_f} \left\|\hat{\mathbf{y}}_t - \mathbf{y}_t\right\|_2 $$

```python
def compute_ade(pred, gt, mask):
    """
    ADE = mean L2 distance over all valid future timesteps.
    """
    pred = _to_numpy(pred)
    gt = _to_numpy(gt)
    mask = _to_numpy(mask)

    diff = pred - gt                      # (N, T, 2)
    dist = np.linalg.norm(diff, axis=-1)  # (N, T)

    num = (dist * mask).sum()             # sum of valid distances
    den = mask.sum() + 1e-8               # count of valid steps
    return float(num / den)
```
This computes exactly what the formula expresses: take the L2 distance at each timestep, filter out padded frames using the mask, and average over all valid predictions.

#### **Final Displacement Error (FDE)**

FDE measures how far off the prediction is at the final valid timestep:

$$ \text{FDE} = \left\|\hat{\mathbf{y}}_{T_f} - \mathbf{y}_{T_f}\right\|_2 $$

```python
def compute_fde(pred, gt, mask):
    """
    FDE = mean L2 distance at the final valid timestep for each sample.
    """
    pred, gt, mask = _to_numpy(pred), _to_numpy(gt), _to_numpy(mask)
    fdes = []

    for i in range(len(pred)):
        valid_idx = np.where(mask[i] > 0)[0]
        if len(valid_idx) == 0:
            continue
        t = valid_idx[-1]                 # final valid timestep
        diff = pred[i, t] - gt[i, t]
        fdes.append(np.linalg.norm(diff))

    return float(np.mean(fdes)) if fdes else 0.0
```


#### **Miss Rate @ 2m (MR@2m)**

A trajectory is counted as a miss if its final predicted point is more than 2 meters away from the ground truth:

$$ \text{MR@2m} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\left[\text{FDE}_i > 2.0\right] $$

```python
def compute_miss_rate(pred, gt, mask, threshold=2.0):
    """
    Miss rate = fraction of trajectories whose final position error > threshold.
    """
    pred, gt, mask = _to_numpy(pred), _to_numpy(gt), _to_numpy(mask)
    N, T, _ = pred.shape
    misses, total = 0, 0

    for i in range(N):
        valid_idx = np.where(mask[i] > 0)[0]
        if len(valid_idx) == 0:
            continue
        t = valid_idx[-1]
        err = np.linalg.norm(pred[i, t] - gt[i, t])
        misses += (err > threshold)
        total += 1

    return float(misses / total) if total > 0 else 0.0
```

---

### Multimodal-Specific Metrics

When evaluating multimodal predictors, standard metrics like ADE and FDE only tell part of the story. If I select the model's most confident prediction and measure its error, I'm testing both trajectory quality *and* confidence calibration simultaneously. To disentangle these, I introduce oracle metrics that measure the best-case performance across all modes, plus calibration metrics that evaluate how well the model knows which mode to trust.

#### **Minimum ADE (minADE)**

Instead of using the model's confidence to select a mode, minADE asks: "If an oracle picked the best mode for each sample, how accurate would the predictions be?" This reveals the model's trajectory generation quality independent of its confidence head:

$$ \text{minADE} = \frac{1}{N} \sum_{i=1}^{N} \min_{m \in [1,M]} \left( \frac{1}{T_f} \sum_{t=1}^{T_f} \left\|\hat{\mathbf{y}}_{i,t}^{(m)} - \mathbf{y}_{i,t}\right\|_2 \right) $$
```python
def compute_min_ade(trajs, gt, mask):
    """
    Oracle minADE: selects the best mode per sample based on actual error.
    
    trajs: (N, M, T, 2) - M trajectory hypotheses per sample
    gt:    (N, T, 2)    - ground truth trajectories
    mask:  (N, T)       - validity mask
    
    Returns: float - average of per-sample minimum ADEs
    """
    N, M, T, _ = trajs.shape
    gt_exp = gt.unsqueeze(1).expand_as(trajs)          # (N, M, T, 2)
    mask_exp = mask.unsqueeze(1).expand(N, M, T)       # (N, M, T)
    
    diff = trajs - gt_exp
    dist = torch.norm(diff, dim=-1) * mask_exp         # (N, M, T)
    
    # ADE per mode: average displacement over valid timesteps
    ade_per_mode = dist.sum(dim=-1) / mask_exp.sum(dim=-1).clamp(min=1)  # (N, M)
    
    # Select minimum across modes for each sample
    return ade_per_mode.min(dim=-1)[0].mean().item()
```

The gap between standard ADE (using confidence-selected mode) and minADE reveals how much performance is lost due to imperfect confidence calibration. A large gap suggests the model generates good trajectories but struggles to identify which one is best.

#### **Minimum FDE (minFDE)**

Similarly, minFDE measures the oracle final displacement:

$$ \text{minFDE} = \frac{1}{N} \sum_{i=1}^{N} \min_{m \in [1,M]} \left\|\hat{\mathbf{y}}_{i,T_f}^{(m)} - \mathbf{y}_{i,T_f}\right\|_2 $$
```python
def compute_min_fde(trajs, gt, mask):
    """
    Oracle minFDE: best mode's final displacement error.
    
    For each sample, finds the mode with smallest final position error,
    regardless of what the confidence head predicted.
    """
    N, M, T, _ = trajs.shape
    valid_counts = mask.sum(dim=-1).long()
    last_idx = (valid_counts - 1).clamp(min=0)          # (N,)
    
    gt_final = gt[torch.arange(N), last_idx]            # (N, 2)
    
    # Compute FDE for each mode
    fde_per_mode = []
    for m in range(M):
        pred_final = trajs[:, m][torch.arange(N), last_idx]  # (N, 2)
        fde_per_mode.append(torch.norm(pred_final - gt_final, dim=-1))
    
    fde_per_mode = torch.stack(fde_per_mode, dim=1)     # (N, M)
    return fde_per_mode.min(dim=-1)[0].mean().item()
```

#### **Minimum Miss Rate (minMR@2m)**

The oracle version of miss rate—what fraction of trajectories would miss even with perfect mode selection:
```python
def compute_min_mr(trajs, gt, mask, threshold=2.0):
    """
    Oracle miss rate: fraction of samples where even the best mode
    has final displacement error > threshold.
    """
    N, M, T, _ = trajs.shape
    valid_counts = mask.sum(dim=-1).long()
    last_idx = (valid_counts - 1).clamp(min=0)
    
    gt_final = gt[torch.arange(N), last_idx]
    
    fde_per_mode = []
    for m in range(M):
        pred_final = trajs[:, m][torch.arange(N), last_idx]
        fde_per_mode.append(torch.norm(pred_final - gt_final, dim=-1))
    
    fde_per_mode = torch.stack(fde_per_mode, dim=1)
    min_fde = fde_per_mode.min(dim=-1)[0]               # (N,)
    
    return (min_fde > threshold).float().mean().item()
```

#### **Confidence Accuracy**

This metric directly measures how often the model's most confident mode is actually the best mode. It answers: "When the model says 'I think mode 3 is most likely,' how often is mode 3 actually closest to ground truth?"

$$ \text{Conf Acc} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\left[\arg\max_m c_i^{(m)} = \arg\min_m E_i^{(m)}\right] $$
```python
def compute_conf_accuracy(trajs, confs, gt, mask):
    """
    Confidence accuracy: fraction of samples where the highest-confidence
    mode is also the mode with lowest ADE.
    
    A perfect confidence head would achieve 100%. Random guessing with
    M=6 modes would achieve ~16.7%.
    """
    N, M, T, _ = trajs.shape
    gt_exp = gt.unsqueeze(1).expand_as(trajs)
    mask_exp = mask.unsqueeze(1).expand(N, M, T)
    
    diff = trajs - gt_exp
    dist = torch.norm(diff, dim=-1) * mask_exp
    ade_per_mode = dist.sum(dim=-1) / mask_exp.sum(dim=-1).clamp(min=1)
    
    best_mode_by_error = ade_per_mode.argmin(dim=-1)    # (N,) - which mode was actually best
    best_mode_by_conf = confs.argmax(dim=-1)            # (N,) - which mode model thought was best
    
    return (best_mode_by_error == best_mode_by_conf).float().mean().item()
```

With M=6 modes, random confidence assignment would yield ~16.7% accuracy. My model achieves ~48.7%, indicating the confidence head has learned meaningful distinctions between modes, though there's room for improvement.

#### **Brier Score**

While confidence accuracy is binary (right or wrong), the Brier score measures how well-calibrated the full probability distribution is:

$$ \text{Brier} = \frac{1}{N} \sum_{i=1}^{N} \sum_{m=1}^{M} \left(c_i^{(m)} - y_i^{(m)}\right)^2 $$

where $$y_i^{(m)} = 1$$ if mode $$m$$ has the lowest error for sample $$i$$, and $$0$$ otherwise.
```python
def compute_brier_score(trajs, confs, gt, mask):
    """
    Brier score for confidence calibration.
    
    Measures the mean squared error between predicted confidence distribution
    and the one-hot "true" distribution (where the best mode gets 1.0).
    
    Lower is better. A perfectly calibrated model placing all confidence
    on the correct mode achieves 0.0. Uniform confidence (1/M each) on
    M=6 modes yields Brier ≈ 0.83.
    """
    N, M, T, _ = trajs.shape
    gt_exp = gt.unsqueeze(1).expand_as(trajs)
    mask_exp = mask.unsqueeze(1).expand(N, M, T)
    
    diff = trajs - gt_exp
    dist = torch.norm(diff, dim=-1) * mask_exp
    ade_per_mode = dist.sum(dim=-1) / mask_exp.sum(dim=-1).clamp(min=1)
    
    best_mode = ade_per_mode.argmin(dim=-1)             # (N,)
    
    # Create one-hot target distribution
    target = torch.zeros_like(confs)                    # (N, M)
    target[torch.arange(N), best_mode] = 1.0
    
    # Brier score: MSE between predicted and target distributions
    return ((confs - target) ** 2).sum(dim=-1).mean().item()
```

My model achieves a Brier score of ~0.63, better than uniform but indicating the confidence probabilities could be sharper and more decisive.

## Baseline Models

Before developing complex neural architectures, I should establish performance baselines using classical kinematic models and a simple learned approach.

### Constant Velocity (CV) Baseline

The simplest physics-based model assumes the agent will continue moving at its current velocity indefinitely. Given the last two valid history positions, we estimate velocity and linearly extrapolate:

$$\mathbf{v} = \frac{\mathbf{p}_{t_2} - \mathbf{p}_{t_1}}{\Delta t \cdot (t_2 - t_1)}$$

$$\hat{\mathbf{p}}_{t+h} = \mathbf{p}_{t_2} + \mathbf{v} \cdot (\Delta t \cdot h), \quad h = 1, \ldots, T_f$$

```python
# baseline.py - Constant Velocity Implementation

DT = 0.1  # Time step between frames (seconds)

def constant_velocity_baseline(history_positions: np.ndarray,
                               future_len: int) -> np.ndarray:
    """
    CV baseline: assume constant velocity equal to the last observed velocity.

    history_positions: (N, H, 2) with NaNs for missing history
    future_len: number of future steps to predict

    Returns:
        preds: (N, future_len, 2)
    """
    N, H, _ = history_positions.shape
    preds = np.zeros((N, future_len, 2), dtype=np.float32)

    for i in range(N):
        hist = history_positions[i]          # (H, 2)
        valid = ~np.isnan(hist).any(axis=1)
        valid_idx = np.where(valid)[0]

        if len(valid_idx) == 0:
            continue  # No history -> predict zeros

        if len(valid_idx) == 1:
            # Only one point -> stay stationary
            last_pos = hist[valid_idx[-1]]
            preds[i] = last_pos
            continue

        # Use last two valid points for velocity estimation
        t1, t2 = valid_idx[-2], valid_idx[-1]
        p1, p2 = hist[t1], hist[t2]

        v = (p2 - p1) / (DT * (t2 - t1))  # velocity vector (m/s)
        last_pos = p2

        for h in range(1, future_len + 1):
            preds[i, h - 1] = last_pos + v * (DT * h)

    return preds
```

### Constant Acceleration (CA) Baseline

A more sophisticated physics model estimates acceleration from the last three valid points:

$$\mathbf{v}_{01} = \frac{\mathbf{p}_1 - \mathbf{p}_0}{\Delta t \cdot (t_1 - t_0)}, \quad \mathbf{v}_{12} = \frac{\mathbf{p}_2 - \mathbf{p}_1}{\Delta t \cdot (t_2 - t_1)}$$

$$\mathbf{a} = \frac{\mathbf{v}_{12} - \mathbf{v}_{01}}{\Delta t \cdot (t_2 - t_0) / 2}$$

$$\hat{\mathbf{p}}_{t+h} = \mathbf{p}_2 + \mathbf{v}_{12} \cdot t + \frac{1}{2}\mathbf{a} \cdot t^2, \quad t = \Delta t \cdot h$$

```python
def constant_acceleration_baseline(history_positions: np.ndarray,
                                   future_len: int) -> np.ndarray:
    """
    CA baseline: estimate constant acceleration from the last three valid points.
    Falls back to CV if insufficient points.
    """
    N, H, _ = history_positions.shape
    preds = np.zeros((N, future_len, 2), dtype=np.float32)

    for i in range(N):
        hist = history_positions[i]
        valid = ~np.isnan(hist).any(axis=1)
        valid_idx = np.where(valid)[0]

        if len(valid_idx) < 3:
            # Fallback to CV behavior
            continue

        # Use last 3 valid points to estimate acceleration
        t0, t1, t2 = valid_idx[-3], valid_idx[-2], valid_idx[-1]
        p0, p1, p2 = hist[t0], hist[t1], hist[t2]

        # Velocities between consecutive points
        v01 = (p1 - p0) / (DT * (t1 - t0))
        v12 = (p2 - p1) / (DT * (t2 - t1))

        # Crude acceleration estimate
        a = (v12 - v01) / (DT * (t2 - t0) / 2.0)
        v_last = v12
        last_pos = p2

        for h in range(1, future_len + 1):
            t = DT * h
            preds[i, h - 1] = last_pos + v_last * t + 0.5 * a * (t ** 2)

    return preds
```

### MLP Baseline

The first learned baseline uses a simple Multi-Layer Perceptron that maps flattened history positions to future trajectory coordinates:

$$\hat{\mathbf{Y}} = \text{MLP}(\text{flatten}(\mathbf{H}))$$

```python
class TrajMLP(nn.Module):
    """
    Simple MLP baseline:
      input: flattened history positions (H * 2)
      output: flattened future positions (T * 2)
    """

    def __init__(self, history_len: int, future_len: int, hidden_dim: int = 256):
        super().__init__()
        in_dim = history_len * 2
        out_dim = future_len * 2

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, H, 2) history positions
        Returns: (N, T, 2) future predictions
        """
        N, H, _ = x.shape
        x_flat = x.reshape(N, H * 2)
        out_flat = self.net(x_flat)
        out = out_flat.reshape(N, -1, 2)
        return out
```

![Baseline Comparison Placeholder]({{ '/assets/images/team36/baseline_compare.png' | relative_url }})
*Fig 3. Comparison of baseline predictions. CV produces straight-line extrapolations, CA adds curvature, and MLP learns smoother patterns but still produces unimodal outputs.*

---

## ResNet Single-Trajectory Model

### Motivation

The baseline models ignore crucial scene context; road geometry, lane markings, and traffic rules that constrain where vehicles can feasibly travel. The ResNet model addresses this by consuming rasterized bird's-eye-view maps alongside agent history.

### Architecture Design

I adapt ResNet-50 for trajectory regression by:

1. **Modifying Input Channels:** replacing the first convolutional layer to accept rasters with additional history channels
2. **Replacing Classification Head:** swapping the 1000=class softmax head with a linear layer predicting $$T_f \times 2$$ coordinates

```python
# train_resnet.py - ResNet Architecture

class LyftResNetModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        history_num_frames = cfg["model_params"]["history_num_frames"]  # 10
        future_num_frames = cfg["model_params"]["future_num_frames"]    # 50

        # Input channels: 3 (RGB satellite) + (history_frames + 1) * 2 (agent positions)
        num_history_channels = (history_num_frames + 1) * 2
        num_in_channels = 3 + num_history_channels  # 3 + 22 = 25

        # Load pretrained ResNet-50
        backbone = resnet50(pretrained=True)

        # Replace first conv to accept custom input channels
        backbone.conv1 = nn.Conv2d(
            num_in_channels,
            backbone.conv1.out_channels,
            kernel_size=backbone.conv1.kernel_size,
            stride=backbone.conv1.stride,
            padding=backbone.conv1.padding,
            bias=False,
        )

        # Replace classification head with trajectory regression
        num_targets = 2 * future_num_frames  # 100 outputs
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_targets)

        self.model = backbone
        self.future_len = future_num_frames

    def forward(self, x):
        out = self.model(x)  # (B, T*2)
        out = out.view(x.shape[0], self.future_len, 2)  # (B, T, 2)
        return out
```

### Loss Function with Temporal Weighting

I use weighted MSE loss that emphasizes later timesteps, which are more critical for planning:

$$\mathcal{L}_{\text{ResNet}} = \frac{\sum_{t=1}^{T_f} w_t \cdot a_t \cdot \|\hat{\mathbf{y}}_t - \mathbf{y}_t\|^2}{\sum_{t=1}^{T_f} w_t \cdot a_t}$$

where $$w_t$$ linearly increases from 0.5 to 2.0 across timesteps, and $$a_t$$ is the availability mask.

```python
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_count = 0.0, 0

    for batch in loader:
        imgs = batch["image"].to(device).float()
        target = batch["target_positions"].to(device).float()      # (B, T, 2)
        avail = batch["target_availabilities"].to(device).float()  # (B, T)

        pred = model(imgs)

        B, T, _ = target.shape
        # Temporal weights: emphasize later timesteps
        weights = torch.linspace(0.5, 2.0, steps=T, device=device)

        diff2 = (pred - target) ** 2                    # (B, T, 2)
        mse_per_step = diff2.sum(dim=-1)                # (B, T)
        mse_per_step = mse_per_step * avail * weights   # Apply masks and weights
        denom = (avail * weights).sum().clamp(min=1e-8)

        loss = mse_per_step.sum() / denom

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total_count += imgs.size(0)

    return total_loss / max(total_count, 1)
```

### Training Configuration

To ensure stable optimization and fast convergence on the 350k-sample Lyft dataset, I will select hyperparameters commonly used for large-scale raster-based trajectory models. A batch size of 64 balances GPU memory usage with gradient stability. Adam with a learning rate of 1e-3 provides rapid early training, while StepLR with γ=0.5 every 3 epochs prevents plateaus and improves late-stage refinement. Ten epochs is sufficient because ResNet-50 is pretrained, and the model only needs to learn the raster-to-trajectory mapping. The raster size is fixed at 224×224 to match the ResNet receptive field and maintain compatibility with the modified backbone.

<table style="width: 100%; border-collapse: collapse; font-size: 16px;">
  <thead>
    <tr>
      <th style="text-align: left; padding: 8px; border-bottom: 2px solid #ccc;">Hyperparameter</th>
      <th style="text-align: left; padding: 8px; border-bottom: 2px solid #ccc;">Value</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">Batch Size</td><td style="padding: 8px; border-bottom: 1px solid #eee;">64</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">Learning Rate</td><td style="padding: 8px; border-bottom: 1px solid #eee;">1e-3</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">Optimizer</td><td style="padding: 8px; border-bottom: 1px solid #eee;">Adam</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">LR Schedule</td><td style="padding: 8px; border-bottom: 1px solid #eee;">StepLR (γ=0.5 every 3 epochs)</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">Training Samples</td><td style="padding: 8px; border-bottom: 1px solid #eee;">350,000</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">Validation Samples</td><td style="padding: 8px; border-bottom: 1px solid #eee;">35,000</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">Epochs</td><td style="padding: 8px; border-bottom: 1px solid #eee;">10</td></tr>
    <tr><td style="padding: 8px;">Input Size</td><td style="padding: 8px;">224 × 224</td></tr>
  </tbody>
</table>


![ResNet Curves Placeholder]({{ '/assets/images/team36/resnet_curves.png' | relative_url }})
*Fig 4. ResNet training curves showing loss convergence and validation metric improvements over 10 epochs.*

---

## Multimodal Raster + Agent History Model

<div style="flex: 1 1 45%; min-width: 300px; text-align: center;">
    <img src="{{ '/assets/images/team36/scene_2286_multimodal.gif' | relative_url }}" alt="Scene 2286 Multimodal Prediction" style="width: 100%; border: 1px solid #ccc; border-radius: 5px;">
    <p><em>Scene 2286 from L5Kit: Comparison on a turn.</em></p>
</div>

### Visualization Compatibility Wrapper

Some visualization code expects a simpler interface: feed in position history, get out trajectories. The `WayformerModel` wrapper provides this interface while internally using the full raster-based model. When no raster is available (e.g., for quick testing), it creates a dummy zero image and relies solely on agent history:
```python
class WayformerModel(nn.Module):
    """
    Wrapper class for visualization compatibility.
    
    Provides a simplified (history, mask) -> (trajectories, confidences) interface.
    When called without raster context, it creates a dummy zero image internally,
    so predictions rely entirely on agent motion history.
    
    This is useful for:
    - Quick testing without setting up the full data pipeline
    - Visualization code that doesn't have access to raster images
    - Comparing history-only vs. history+raster performance
    
    Note: Predictions will be less accurate without raster context, as the
    model can't see road geometry, lane markings, or other scene features.
    """

    def __init__(self, cfg: Dict, in_channels: int = 25):
        super().__init__()
        self.cfg = cfg
        self.model = RasterTrajectoryModel(cfg, in_channels=in_channels)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simplified forward pass using only position history.
        
        Args:
            x: (B, H, 2) history positions in agent-local coordinates
            mask: (B, H) boolean mask (True = valid observation)
        
        Returns:
            trajectories: (B, M, T, 2) predicted future positions
            confidences: (B, M) probability distribution over modes
        """
        device = x.device
        B, H, _ = x.shape
        mp = self.cfg["model_params"]

        H_conf = mp["history_num_frames"]
        T = mp["future_num_frames"]

        # Convert positions to [x, y, vx, vy] format
        x_np = x.detach().cpu().numpy()
        hist_pos = np.nan_to_num(x_np, nan=0.0)
        
        # Compute velocities from position differences
        if H >= 2:
            vel = np.zeros_like(hist_pos)
            vel[:, 1:, :] = (hist_pos[:, 1:, :] - hist_pos[:, :-1, :]) / mp["step_time"]
        else:
            vel = np.zeros_like(hist_pos)

        agent_hist = np.concatenate([hist_pos, vel], axis=-1)  # (B, H, 4)

        # Pad or truncate to expected history length
        if H < H_conf:
            pad = np.zeros((B, H_conf - H, 4), dtype=np.float32)
            agent_hist = np.concatenate([pad, agent_hist], axis=1)
        else:
            agent_hist = agent_hist[:, -H_conf:, :]

        agent_hist = torch.from_numpy(agent_hist).float().to(device)
        hist_mask = torch.zeros(B, H_conf, dtype=torch.bool, device=device)

        # Create dummy zero image (no raster context)
        # The model will rely entirely on agent history for predictions
        C = self.model.cnn[0].weight.shape[1]  # Get expected channels from model
        img = torch.zeros(B, C, 224, 224, device=device)

        batch = {
            "image": img,
            "agent_hist": agent_hist,
            "agent_hist_mask": hist_mask,
            "target": torch.zeros(B, T, 2, device=device),
            "avail": torch.ones(B, T, device=device),
        }

        pred = self.model(batch)
        return pred["trajectories"], pred["confidences"]

    def load_from_checkpoint(self, path: str, device: torch.device):
        """Load model weights from a training checkpoint."""
        ckpt = torch.load(path, map_location=device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        return self
```

This wrapper is intentionally limited. Without raster context, the model is essentially blind to road geometry. It's meant for quick experiments and visualization, not production inference. For best results, always use the full `RasterTrajectoryModel` with proper raster images.

### The Need for Multimodality

A fundamental limitation of the ResNet model is that it produces a single deterministic trajectory. Real driving scenarios are inherently uncertain. At an intersection, a vehicle might turn left, right, or go straight with roughly equal probability. A single prediction cannot capture this uncertainty and may produce an "averaged" trajectory that lies between the true modes, satisfying none of them.

Our final model addresses this by predicting **M = 6 diverse trajectory hypotheses** with associated confidence scores, enabling downstream planners to reason about multiple possible futures.

## Model and Design Rationale

To build an effective trajectory-prediction model, I evaluated several architectural options used in recent motion-forecasting literature. The final design reflects a balance of prediction accuracy, computational efficiency, and compatibility with rasterized map inputs.

The core idea is that future motion depends on two complementary factors:

1. **Scene context** like road geometry, lane direction, obstacles, sidewalks, intersections  
2. **Agent dynamics** such as recent motion patterns such as speed, heading changes, and acceleration

No single modality captures both. The architecture therefore merges visual map features and temporal agent features in a lightweight but expressive way.

#### **CNN Backbone (Spatial Encoding).**  
Rasterized semantic maps behave like images, and convolutional networks extract spatial patterns efficiently. Alternatives such as Vision Transformers or graph networks were considered, but they introduce significantly higher computational cost and require much larger datasets to train effectively. A compact 4-layer CNN offers a good trade-off: fast training, stable gradients, and strong performance on structured map inputs.

#### **Agent-History MLP (Motion Encoding).**  
The history vector is only `[x, y, vx, vy] × 10` frames. This is low-dimensional and does not need a heavy recurrent model. I chose a 2-layer MLP because it captures short-term trends (slowing, turning, drifting) at minimal cost. More complex sequence models (GRUs, temporal transformers) were tested but added overhead with little improvement for this representation.

#### **Feature Fusion (Context × Dynamics).**  
The model must interpret agent motion *within* the context of the environment. Fusing the CNN embedding with the history embedding lets the network learn combined features such as deceleration toward an intersection, or drifting toward a lane boundary.

#### **Multi-Modal Prediction Head (M = 6).**  
Future motion is uncertain, especially at intersections. A single deterministic trajectory collapses multiple possible futures into an unrealistic average. Multi-modal heads were chosen over mixture-density networks or CVAE-based sampling because they:
- produce consistent, interpretable modes  
- train stably  
- are easy for downstream planners to consume  

Six modes (M = 6) gave the best balance between coverage of diverse futures and computational cost.

Together, these design choices yield a model that is fast to train, easy to interpret, and capable of predicting diverse, realistic futures in complex urban environments.

### Architecture Overview

The architecture consists of four main components:

1. **CNN Backbone:** Processes semantic raster maps to extract spatial features
2. **History Encoder MLP:** Encodes agent motion history (position + velocity)
3. **Fusion Module:** Combines visual and motion features
4. **Prediction Heads:** Output multiple trajectories and confidence distribution

![Multimodal Architecture Placeholder]({{ '/assets/images/team36/model_architecture.png' | relative_url }})
*Fig 5. Multimodal model architecture. The CNN backbone processes rasterized scene context while the History MLP encodes agent dynamics. Fused features are decoded into M=6 trajectory hypotheses with confidence scores.*

### Detailed Architecture Implementation

```python
class RasterTrajectoryModel(nn.Module):
    """
    CNN (semantic raster) + MLP (agent history) -> multi-modal trajectory.
    
    Architecture:
    - CNN backbone: 4-layer ConvNet with BatchNorm, outputs 256-dim feature
    - History MLP: 2-layer MLP encoding [x, y, vx, vy] × 10 frames
    - Fusion: 2-layer MLP combining CNN + history features
    - Trajectory Head: Linear layer outputting M × T × 2 coordinates
    - Confidence Head: Linear layer outputting M logits → softmax
    """

    def __init__(self, cfg: Dict, in_channels: int = 3):
        super().__init__()
        mp = cfg["model_params"]
        arch = cfg["model_arch"]

        self.history_len = mp["history_num_frames"]  # 10
        self.future_len = mp["future_num_frames"]    # 50
        self.n_modes = arch["n_modes"]               # 6
        d_model = arch["d_model"]                    # 256
        dropout = arch["dropout"]                    # 0.1

        # ============ CNN BACKBONE ============
        # Lightweight 4-layer ConvNet for fast training
        self.cnn = nn.Sequential(
            # Layer 1: 224 -> 112
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Layer 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Layer 4: 28 -> 14 -> 1 (global pool)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
        )
        self.cnn_out_dim = 256

        # ============ HISTORY ENCODER ============
        # MLP over flattened [x, y, vx, vy] × history_len
        self.hist_mlp = nn.Sequential(
            nn.Linear(self.history_len * 4, d_model),  # 40 -> 256
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),               # 256 -> 256
            nn.ReLU(inplace=True),
        )

        # ============ FUSION MODULE ============
        self.fusion = nn.Sequential(
            nn.Linear(self.cnn_out_dim + d_model, d_model),  # 512 -> 256
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),                     # 256 -> 256
            nn.ReLU(inplace=True),
        )

        # ============ PREDICTION HEADS ============
        # Trajectory head: outputs M modes × T timesteps × 2 coordinates
        self.out_traj = nn.Linear(d_model, self.n_modes * self.future_len * 2)
        # Confidence head: outputs M mode logits
        self.out_conf = nn.Linear(d_model, self.n_modes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        img = batch["image"]        # (B, C, H, W)
        hist = batch["agent_hist"]  # (B, H, 4) where 4 = [x, y, vx, vy]

        B = img.shape[0]

        # Extract CNN features from raster
        feat_img = self.cnn(img)           # (B, 256, 1, 1)
        feat_img = feat_img.view(B, -1)    # (B, 256)

        # Extract history features
        feat_hist = hist.view(B, -1)       # (B, H*4) = (B, 40)
        feat_hist = self.hist_mlp(feat_hist)  # (B, 256)

        # Fuse visual and motion features
        fused = self.fusion(torch.cat([feat_img, feat_hist], dim=-1))  # (B, 256)

        # Generate outputs
        traj_flat = self.out_traj(fused)       # (B, M*T*2)
        confid_logits = self.out_conf(fused)   # (B, M)

        # Reshape trajectory output
        traj = traj_flat.view(B, self.n_modes, self.future_len, 2)  # (B, M, T, 2)
        # Convert confidence logits to probability distribution
        confid = F.softmax(confid_logits, dim=-1)                   # (B, M)

        return {
            "trajectories": traj,
            "confidences": confid,           # probs (for eval)
            "confid_logits": confid_logits,  # logits (for loss)
        }
```

## Data Pipeline

We need a system that converts the raw **Lyft L5Kit** samples into the exact tensors the model can train on. In this project, the dataset does not directly provide the features the model needs, so the pipeline builds them step-by-step.

It begins with the **rasterized map image** from **L5Kit** and **normalizes** it so the **CNN** can process it. It then takes the agent’s **past positions** and computes **velocities** from frame-to-frame differences, forming the **motion-history vector** `[x, y, vx, vy]` for each timestep. Because different agents have different amounts of valid history, the pipeline **pads or trims** the sequence so every sample has a **fixed history length**, ensuring consistent input size during batching.

Finally, it extracts the **future positions** from the dataset. These become the model’s **regression targets**.

By the time a batch reaches the network, each item has been transformed into a uniform structure:

- a **normalized raster** for the **CNN**  
- a **fixed-length motion-history tensor** for the **MLP**  
- **future trajectories** for the loss function  

This preprocessing step is what allows the model to treat every sample identically, **train efficiently**, and learn both **scene context** and **agent dynamics** from the raw dataset.


```python
class RasterLyftDataset(Dataset):
    """
    Wraps L5Kit AgentDataset, returning:
      - image: semantic raster (C, H, W)
      - agent_hist: (H, 4) containing [x, y, vx, vy] per timestep
      - agent_hist_mask: (H,) boolean mask for padding
      - target: (T, 2) future positions
      - avail: (T,) availability mask
    """

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.agent_dataset[idx]

        # 1) Normalize raster image
        img = data["image"].astype(np.float32) / 255.0  # (C, H, W)

        # 2) Compute velocity from position differences
        hist_pos = data["history_positions"].astype(np.float32)
        hist_pos = np.nan_to_num(hist_pos, nan=0.0)
        H0 = hist_pos.shape[0]

        if H0 >= 2:
            vel = np.zeros_like(hist_pos)
            vel[1:] = (hist_pos[1:] - hist_pos[:-1]) / self.mp["step_time"]
        else:
            vel = np.zeros_like(hist_pos)

        # Concatenate position and velocity: [x, y, vx, vy]
        agent_hist = np.concatenate([hist_pos, vel], axis=-1)  # (H0, 4)

        # Pad or truncate to fixed history length
        if H0 < self.history_len:
            pad = np.zeros((self.history_len - H0, 4), dtype=np.float32)
            agent_hist = np.concatenate([pad, agent_hist], axis=0)
        else:
            agent_hist = agent_hist[-self.history_len:]

        # 3) Prepare target and availability
        target = data["target_positions"].astype(np.float32)
        avail = data["target_availabilities"].astype(np.float32)

        return {
            "image": torch.from_numpy(img),
            "agent_hist": torch.from_numpy(agent_hist),
            "target": torch.from_numpy(target),
            "avail": torch.from_numpy(avail),
        }
```

---

## Loss Function Derivation

### The Challenge of Multimodal Learning

Training multimodal predictors presents a fundamental challenge: which of the M predicted modes should match the single ground truth trajectory? A naive approach of averaging loss across all modes would cause all predictions to collapse to the mean. We need a loss function that:

1. Encourages at least one mode to match the ground truth well
2. Encourages diversity among modes
3. Trains the confidence head to predict which mode is best

### Soft Winner-Takes-All Loss

We implement a **soft winner-takes-all (WTA)** loss that allows gradients to flow primarily to the mode closest to the ground truth while maintaining diversity:

#### **Step 1: Compute per-mode errors**

For each mode $$m$$, compute the total L2 error:

$$E_m = \sum_{t=1}^{T_f} a_t \cdot \|\hat{\mathbf{y}}_t^{(m)} - \mathbf{y}_t\|_2$$

where $$a_t$$ is the availability mask.

#### **Step 2: Soft mode selection**

Instead of hard selection (which prevents gradient flow to non-winning modes), we use temperature-scaled softmax:

$$w_m = \frac{\exp(-E_m / \tau)}{\sum_{m'} \exp(-E_{m'} / \tau)}$$

where $$\tau = 0.1$$ is a temperature parameter. Lower $$\tau$$ makes the weighting sharper.

#### **Step 3: Regression loss**

The weighted regression loss becomes:

$$\mathcal{L}_{\text{reg}} = \sum_{m=1}^{M} w_m \cdot E_m$$

#### **Step 4: Confidence loss**

We train the confidence head to predict which mode is best using cross-entropy:

$$m^* = \arg\min_m E_m$$

$$\mathcal{L}_{\text{conf}} = -\log(c_{m^*})$$

#### **Step 5: Total loss**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{reg}} + \lambda_{\text{conf}} \cdot \mathcal{L}_{\text{conf}}$$

where $$\lambda_{\text{conf}} = 0.5$$.

```python
class MultiModalTrajectoryLoss(nn.Module):
    """
    Multi-modal L2 loss with soft winner-takes-all + confidence cross-entropy.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.n_modes = cfg["model_arch"]["n_modes"]
        self.tau = 0.1          # Temperature for soft WTA
        self.lambda_conf = 0.5  # Weight for confidence loss

    def forward(
        self, pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        traj = pred["trajectories"]   # (B, M, T, 2)
        confid = pred["confidences"]  # (B, M)

        gt = batch["target"]          # (B, T, 2)
        avail = batch["avail"]        # (B, T)

        B, M, T, _ = traj.shape

        # Expand ground truth to match trajectory shape
        gt_exp = gt.unsqueeze(1).expand_as(traj)         # (B, M, T, 2)
        avail_exp = avail.unsqueeze(1).expand(B, M, T)   # (B, M, T)

        # Compute L2 distances per mode per timestep
        diff = traj - gt_exp
        dist = torch.norm(diff, dim=-1) * avail_exp      # (B, M, T)

        # Sum over timesteps to get per-mode error
        mode_errors = dist.sum(dim=-1)  # (B, M)

        # Find best mode (for confidence supervision)
        with torch.no_grad():
            _, best_mode = mode_errors.min(dim=-1)  # (B,)

        # Soft WTA weights (temperature-scaled softmax)
        w = F.softmax(-mode_errors / self.tau, dim=-1)  # (B, M)
        reg_loss = (w * mode_errors).sum(dim=-1).mean()

        # Cross-entropy loss for confidence head
        confid_logits = pred["confid_logits"]  # (B, M)
        conf_loss = F.cross_entropy(confid_logits, best_mode)

        # ===== Compute ADE/FDE for logging =====
        best_traj = traj[torch.arange(B, device=traj.device), best_mode]  # (B, T, 2)
        valid_counts = avail.sum(dim=-1).clamp(min=1)
        last_idx = (avail.cumsum(dim=-1) == valid_counts.unsqueeze(-1)).float().argmax(dim=-1)

        fde = torch.norm(
            best_traj[torch.arange(B), last_idx] - gt[torch.arange(B), last_idx],
            dim=-1,
        ).mean()

        # ADE over best mode only
        best_dist = torch.norm(best_traj - gt, dim=-1) * avail  # (B, T)
        ade = best_dist.sum() / avail.sum().clamp(min=1)

        total = reg_loss + self.lambda_conf * conf_loss

        return {
            "total": total,
            "reg": reg_loss,
            "conf": conf_loss,
            "ade": ade,
            "fde": fde,
        }
```

---

## Training Procedure

The configuration is designed around the spatial and temporal characteristics of urban driving while keeping training efficient. A history length of `10 frames` at `10 Hz` captures `1 second` of recent motion, which is enough to infer short-term intent such as braking, accelerating, or initiating a turn. Predicting `50 future frames` covers a `5-second horizon`, which is the standard window used in motion-forecasting benchmarks. A `224×224 raster` at `0.5 m/pixel` yields a `~112 m` field of view, which is large enough to include intersections, road geometry, and nearby agents. The ego-center offset shifts the raster forward so the model sees more of the road ahead, where the agent’s future motion depends most.

A batch size of `128` provides stable gradients without exceeding GPU memory. Only a few epochs are needed because the dataset is extremely large and each epoch covers hundreds of thousands of samples. A learning rate of `1e-3` with weight decay of `1e-4` is a well-behaved optimization setup for `Adam` on mid-sized networks. The architectural choices of `256-dim embeddings`, `6 trajectory modes`, and `0.1 dropout` strike a balance between capacity, regularization, and training speed, enabling the model to learn diverse, realistic futures without overfitting.


### Training Configuration

```python
def build_cfg():
    return {
        "format_version": 4,
        "model_params": {
            "history_num_frames": 10,    # 1 second of history
            "future_num_frames": 50,     # 5 seconds of future
            "step_time": 0.1,            # 10 Hz
            "render_ego_history": True,
            "render_ego_center": True,
        },
        "raster_params": {
            "raster_size": [224, 224],
            "pixel_size": [0.5, 0.5],    # meters per pixel
            "ego_center": [0.25, 0.5],   # ego position in raster
            "map_type": "py_semantic",   # semantic map (road semantics)
        },
        "train_data_loader": {
            "batch_size": 128,
            "shuffle": True,
            "num_workers": 0,
        },
        "training": {
            "num_epochs": 3,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
        },
        "model_arch": {
            "d_model": 256,
            "n_modes": 6,
            "dropout": 0.1,
        },
    }
```

### Training Loop with Mixed Precision

We can employ several optimization techniques for efficient training:

1.  **Mixed Precision Training:** Using `float16` for forward/backward passes to reduce memory and accelerate computation.
2.  **Gradient Clipping:** Preventing exploding gradients (`max norm = 5.0`).
3.  **AdamW Optimizer:** Uses `weight decay` regularization for better generalization.
4.  **Cosine Annealing LR:** Smooth learning rate decay over `num_epochs`.
5.  **Best Checkpoint Saving:** Based on minimizing `validation ADE`.

```python
def train_model(resume_path: Optional[str] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = build_cfg()
    tp = cfg["training"]

    # Initialize model, loss, optimizer
    model = RasterTrajectoryModel(cfg, in_channels=in_channels).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = MultiModalTrajectoryLoss(cfg)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=tp["learning_rate"],
        weight_decay=tp["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tp["num_epochs"], eta_min=1e-5
    )
    scaler = GradScaler()  # For mixed precision

    best_val_ade = float("inf")

    for epoch in range(num_epochs):
        # ========== TRAINING ==========
        model.train()
        train_losses = {"total": 0, "reg": 0, "conf": 0, "ade": 0, "fde": 0}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            # Mixed precision forward pass
            with autocast(dtype=torch.float16):
                pred = model(batch)
                losses = criterion(pred, batch)
                loss_total = losses["total"]

            # Scaled backward pass
            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # Gradient clipping
            scaler.step(optimizer)
            scaler.update()

            for k in train_losses:
                train_losses[k] += losses[k].item()

            pbar.set_postfix({
                "loss": f"{losses['total'].item():.4f}",
                "ade": f"{losses['ade'].item():.2f}",
                "fde": f"{losses['fde'].item():.2f}",
            })

        # ========== VALIDATION ==========
        model.eval()
        val_losses = {"total": 0, "reg": 0, "conf": 0, "ade": 0, "fde": 0}

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                pred = model(batch)
                losses = criterion(pred, batch)

                for k in val_losses:
                    val_losses[k] += losses[k].item()

        scheduler.step()

        # Save best checkpoint
        if val_losses["ade"] / len(val_loader) < best_val_ade:
            best_val_ade = val_losses["ade"] / len(val_loader)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_ade": best_val_ade,
                "cfg": cfg,
            }, "best_raster_model.pth")

    return model
```

### Training Metrics Summary

![Multimodal Curves Placeholder]({{ '/assets/images/team36/multimodal_training_curves.png' | relative_url }})
*Fig 6. Training and validation curves for the multimodal model showing loss convergence, ADE/FDE improvements, and learning rate schedule.*

---

## Quantitative Results

### Final Model Comparison

<div style="width: 100%; display: flex; justify-content: center;">
<table style="width: 95%; border-collapse: collapse; border: 1px solid #ddd; margin: 20px 0;">
  <thead>
    <tr style="border-bottom: 2px solid #333; background-color: #f6f8fa;">
      <th style="padding: 12px; text-align: left;">Model</th>
      <th style="padding: 12px; text-align: left;">Modalities</th>
      <th style="padding: 12px; text-align: left;">ADE ↓</th>
      <th style="padding: 12px; text-align: left;">FDE ↓</th>
      <th style="padding: 12px; text-align: left;">MR@2m ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 10px;">Constant Velocity</td>
      <td style="padding: 10px;"><code>1</code></td>
      <td style="padding: 10px;"><code>8.42</code></td>
      <td style="padding: 10px;"><code>15.67</code></td>
      <td style="padding: 10px;"><code>89.0%</code></td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 10px;">Constant Acceleration</td>
      <td style="padding: 10px;"><code>1</code></td>
      <td style="padding: 10px;"><code>7.81</code></td>
      <td style="padding: 10px;"><code>14.23</code></td>
      <td style="padding: 10px;"><code>85.0%</code></td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 10px;">MLP Baseline</td>
      <td style="padding: 10px;"><code>1</code></td>
      <td style="padding: 10px;"><code>5.34</code></td>
      <td style="padding: 10px;"><code>9.87</code></td>
      <td style="padding: 10px;"><code>72.0%</code></td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 10px;">ResNet Single-Traj</td>
      <td style="padding: 10px;"><code>1</code></td>
      <td style="padding: 10px;"><code>3.12</code></td>
      <td style="padding: 10px;"><code>5.89</code></td>
      <td style="padding: 10px;"><code>48.0%</code></td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 10px;">Multimodal (by confidence)</td>
      <td style="padding: 10px;"><code>6</code></td>
      <td style="padding: 10px;"><code>1.10</code></td>
      <td style="padding: 10px;"><code>2.40</code></td>
      <td style="padding: 10px;"><code>27.2%</code></td>
    </tr>
    <tr>
      <td style="padding: 10px;">Multimodal (oracle best)</td>
      <td style="padding: 10px;"><code>6</code></td>
      <td style="padding: 10px;"><code>0.45</code></td>
      <td style="padding: 10px;"><code>0.95</code></td>
      <td style="padding: 10px;"><code>13.4%</code></td>
    </tr>
  </tbody>
</table>
</div>

### Multimodal Model Detailed Metrics

<div style="width: 100%; display: flex; justify-content: center;">
<table style="width: 95%; border-collapse: collapse; border: 1px solid #ddd; margin: 20px 0;">
  <thead>
    <tr style="border-bottom: 2px solid #333; background-color: #f6f8fa;">
      <th style="padding: 12px; text-align: left;">Metric</th>
      <th style="padding: 12px; text-align: left;">Value</th>
      <th style="padding: 12px; text-align: left;">Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 10px;">ADE (by confidence)</td>
      <td style="padding: 10px;"><code>1.10 m</code></td>
      <td style="padding: 10px;">Error using model's highest-confidence mode</td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 10px;">FDE (by confidence)</td>
      <td style="padding: 10px;"><code>2.40 m</code></td>
      <td style="padding: 10px;">Final position error using model's choice</td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 10px;">minADE (oracle)</td>
      <td style="padding: 10px;"><code>0.45 m</code></td>
      <td style="padding: 10px;">Best possible with perfect mode selection</td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 10px;">minFDE (oracle)</td>
      <td style="padding: 10px;"><code>0.95 m</code></td>
      <td style="padding: 10px;">Best possible final error across all modes</td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 10px;">Confidence Accuracy</td>
      <td style="padding: 10px;"><code>48.7%</code></td>
      <td style="padding: 10px;">How often the model picks the best mode</td>
    </tr>
    <tr>
      <td style="padding: 10px;">Brier Score</td>
      <td style="padding: 10px;"><code>0.626</code></td>
      <td style="padding: 10px;">Confidence calibration quality (lower = better)</td>
    </tr>
  </tbody>
</table>
</div>

### Trajectory Behavior Analysis

Not all trajectories are equally difficult to predict. A vehicle traveling straight on a highway is far more predictable than one navigating a complex intersection. To understand where my model excels and struggles, I classify trajectories by their cumulative curvature and analyze performance separately.

#### Curvature-Based Classification

I compute the total angular change along each ground-truth trajectory. Trajectories with cumulative curvature below a threshold (0.3 radians ≈ 17°) are classified as "straight," while those above are "turning":
```python
def compute_turn_metrics(trajs, confs, gt, mask, turn_threshold=0.3):
    """
    Classifies trajectories as straight vs turning based on cumulative
    angular change, then computes metrics separately for each category.
    
    The curvature is computed by summing the absolute angle between
    consecutive velocity vectors along the ground-truth path.
    
    Args:
        trajs: (N, M, T, 2) predicted trajectories
        confs: (N, M) confidence scores
        gt: (N, T, 2) ground truth
        mask: (N, T) validity mask
        turn_threshold: radians of cumulative curvature to classify as turn
    
    Returns:
        dict with counts and per-category metrics
    """
    N, M, T, _ = trajs.shape
    gt_np = gt.cpu().numpy()
    mask_np = mask.cpu().numpy()
    
    curvatures = []
    for i in range(N):
        total_angle = 0.0
        valid_t = int(mask_np[i].sum())
        
        if valid_t >= 3:
            for t in range(1, valid_t - 1):
                # Velocity vectors between consecutive points
                v1 = gt_np[i, t] - gt_np[i, t-1]
                v2 = gt_np[i, t+1] - gt_np[i, t]
                
                # Signed angle between vectors using cross and dot products
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                dot = v1[0] * v2[0] + v1[1] * v2[1]
                total_angle += abs(np.arctan2(cross, dot + 1e-6))
        
        curvatures.append(total_angle)
    
    curvatures = np.array(curvatures)
    is_turn = curvatures > turn_threshold
    
    n_straight = (~is_turn).sum()
    n_turn = is_turn.sum()
    
    results = {
        "n_straight": int(n_straight),
        "n_turn": int(n_turn)
    }
    
    # Compute metrics for straight trajectories
    if n_straight > 0:
        straight_idx = np.where(~is_turn)[0]
        results["straight_minADE"] = compute_min_ade(
            trajs[straight_idx], gt[straight_idx], mask[straight_idx])
        results["straight_conf_acc"] = compute_conf_accuracy(
            trajs[straight_idx], confs[straight_idx], 
            gt[straight_idx], mask[straight_idx])
    
    # Compute metrics for turning trajectories
    if n_turn > 0:
        turn_idx = np.where(is_turn)[0]
        results["turn_minADE"] = compute_min_ade(
            trajs[turn_idx], gt[turn_idx], mask[turn_idx])
        results["turn_conf_acc"] = compute_conf_accuracy(
            trajs[turn_idx], confs[turn_idx], 
            gt[turn_idx], mask[turn_idx])
    
    return results
```

#### Why This Matters

The breakdown reveals an interesting asymmetry in model behavior. Straight trajectories are predicted with exceptional accuracy (minADE ~0.17m) because they have low inherent uncertainty. There's really only one plausible future. However, confidence accuracy on straight trajectories is notably poor (~21.8%), suggesting the model hedges its bets even when it shouldn't.

Turning trajectories, which make up the vast majority of the validation set (~95%), show higher minADE (~0.46m) reflecting their greater complexity, but substantially better confidence accuracy (~50%). This indicates the model has learned to distinguish between different turning behaviors (sharp left vs. gentle curve vs. lane change) and assigns confidence accordingly.

This analysis suggests a potential improvement: the model could benefit from learning to recognize when a scenario is "simple" (straight road, no intersection) and concentrate confidence on a single mode, rather than always hedging across multiple hypotheses.

### Analysis

The progression from physics-based baselines to our multimodal architecture demonstrates the cumulative value of each design decision. The Constant Velocity baseline achieves `8.42m ADE` by simply extrapolating the last observed velocity, failing catastrophically when agents accelerate, brake, or turn. Adding acceleration modeling in the CA baseline provides modest improvement to `7.81m ADE`, but still cannot capture the complex dynamics of real driving behavior.

The transition to learned approaches marks a significant turning point. Our MLP baseline, despite having no scene context, reduces ADE to `5.34m` by learning statistical patterns from the training distribution. The ResNet model then demonstrates the critical importance of visual scene understanding, it achieves `3.12m ADE`, a `41% improvement` over the MLP. This confirms that road geometry, lane markings, and intersection structure provide essential constraints on feasible future trajectories.

Our multimodal architecture achieves the best performance across all metrics, with `1.10m ADE` using the model's confidence-selected mode. More revealing is the oracle analysis: when we select the best of the six predicted modes for each sample, ADE drops to just `0.45m` and miss rate falls to `13.4%`. This `59% gap` between confidence-selected and oracle performance indicates that the model generates good trajectory hypotheses but has significant room for improvement in confidence calibration.

### Trajectory Type Breakdown

<div style="width: 100%; display: flex; justify-content: center;">
<table style="width: 95%; border-collapse: collapse; border: 1px solid #ddd; margin: 20px 0;">
  <thead>
    <tr style="border-bottom: 2px solid #333; background-color: #f6f8fa;">
      <th style="padding: 12px; text-align: left;">Category</th>
      <th style="padding: 12px; text-align: left;">Samples</th>
      <th style="padding: 12px; text-align: left;">minADE</th>
      <th style="padding: 12px; text-align: left;">Confidence Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 10px;">Straight</td>
      <td style="padding: 10px;"><code>998</code> (<code>4.6%</code>)</td>
      <td style="padding: 10px;"><code>0.17 m</code></td>
      <td style="padding: 10px;"><code>21.8%</code></td>
    </tr>
    <tr>
      <td style="padding: 10px;">Turning</td>
      <td style="padding: 10px;"><code>20,627</code> (<code>95.4%</code>)</td>
      <td style="padding: 10px;"><code>0.46 m</code></td>
      <td style="padding: 10px;"><code>50.0%</code></td>
    </tr>
  </tbody>
</table>
</div>

The turn versus straight breakdown reveals interesting patterns in model behavior. Straight trajectories are predicted with exceptional accuracy, which is expected given their lower inherent uncertainty. However, confidence accuracy on straight trajectories is notably poor at <code>21.8%</code>, suggesting the model struggles to identify when a trajectory will be straight versus when it might curve.

Turning trajectories, which comprise <code>95.4%</code> of our evaluation set, show higher minADE (<code>0.46m</code>) reflecting their greater complexity, but substantially better confidence accuracy at <code>50.0%</code>. This indicates the model has learned meaningful distinctions between different turning behaviors.

### Oracle Performance Gap

The gap between confidence-selected metrics and oracle metrics represents the potential improvement achievable through better confidence calibration alone:

<div style="width: 100%; display: flex; justify-content: center;">
<table style="width: 95%; border-collapse: collapse; border: 1px solid #ddd; margin: 20px 0;">
  <thead>
    <tr style="border-bottom: 2px solid #333; background-color: #f6f8fa;">
      <th style="padding: 12px; text-align: left;">Metric</th>
      <th style="padding: 12px; text-align: left;">By Confidence</th>
      <th style="padding: 12px; text-align: left;">Oracle</th>
      <th style="padding: 12px; text-align: left;">Potential Improvement</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 10px;">ADE</td>
      <td style="padding: 10px;"><code>1.10 m</code></td>
      <td style="padding: 10px;"><code>0.45 m</code></td>
      <td style="padding: 10px;"><code>59%</code> reduction possible</td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 10px;">FDE</td>
      <td style="padding: 10px;"><code>2.40 m</code></td>
      <td style="padding: 10px;"><code>0.95 m</code></td>
      <td style="padding: 10px;"><code>60%</code> reduction possible</td>
    </tr>
    <tr>
      <td style="padding: 10px;">MR@2m</td>
      <td style="padding: 10px;"><code>27.2%</code></td>
      <td style="padding: 10px;"><code>13.4%</code></td>
      <td style="padding: 10px;"><code>51%</code> reduction possible</td>
    </tr>
  </tbody>
</table>
</div>

## Inference Utilities

Before visualizing predictions, I need utilities that convert raw dataset samples into the format expected by each model. These functions bridge the gap between L5Kit's data format and my model's input requirements.

### Building Raster Batches for Inference

The multimodal model expects a specific input dictionary with normalized images, velocity-augmented history, and properly shaped tensors. This function handles all the preprocessing:
```python
def build_raster_batch_from_sample(data, cfg, device):
    """
    Converts a single L5Kit sample into the batch format expected by
    RasterTrajectoryModel.
    
    This replicates the preprocessing done by RasterLyftDataset but for
    a single sample during inference/visualization.
    
    Args:
        data: dict from AgentDataset or EgoDataset containing:
            - "image": (C, H, W) uint8 raster image
            - "history_positions": (H0, 2) past positions, may have NaN
            - "history_availabilities": (H0,) optional validity mask
        cfg: model configuration dict
        device: torch device for output tensors
    
    Returns:
        batch: dict with keys matching model's forward() signature
    """
    mp = cfg["model_params"]
    history_len = mp["history_num_frames"]
    step_time = mp["step_time"]
    future_len = mp["future_num_frames"]

    # 1) Normalize image: [0, 255] -> [0, 1]
    img = data["image"].astype(np.float32) / 255.0      # (C, H, W)
    img_t = torch.from_numpy(img).unsqueeze(0).to(device)  # (1, C, H, W)

    # 2) Process history positions and compute velocities
    hist_pos = data["history_positions"].astype(np.float32)  # (H0, 2)
    hist_pos = np.nan_to_num(hist_pos, nan=0.0)
    H0 = hist_pos.shape[0]

    # Velocity = position difference / time step
    if H0 >= 2:
        vel = np.zeros_like(hist_pos)
        vel[1:] = (hist_pos[1:] - hist_pos[:-1]) / step_time
    else:
        vel = np.zeros_like(hist_pos)

    # Concatenate position and velocity: [x, y, vx, vy]
    agent_hist = np.concatenate([hist_pos, vel], axis=-1)  # (H0, 4)

    # 3) Pad or truncate to fixed history length
    if H0 < history_len:
        # Pad at the beginning (older timesteps)
        pad = np.zeros((history_len - H0, 4), dtype=np.float32)
        agent_hist = np.concatenate([pad, agent_hist], axis=0)
    else:
        # Keep most recent frames
        agent_hist = agent_hist[-history_len:]

    agent_hist_t = torch.from_numpy(agent_hist).unsqueeze(0).to(device)

    # 4) Build history availability mask
    # True = invalid/padded, False = valid observation
    hist_avail = data.get("history_availabilities",
                          np.ones(H0, dtype=np.float32)).astype(bool)
    if H0 < history_len:
        pad_mask = np.zeros(history_len - H0, dtype=bool)
        hist_mask = np.concatenate([~pad_mask, ~hist_avail], axis=0)
    else:
        hist_mask = ~hist_avail[-history_len:]

    hist_mask_t = torch.from_numpy(hist_mask).unsqueeze(0).to(device)

    # 5) Dummy target tensors (not used during inference, but required by forward())
    target_t = torch.zeros(1, future_len, 2, device=device)
    avail_t = torch.ones(1, future_len, device=device)

    return {
        "image": img_t,
        "agent_hist": agent_hist_t,
        "agent_hist_mask": hist_mask_t,
        "target": target_t,
        "avail": avail_t,
    }
```

### Extracting All Mode Predictions

For visualization, I often want to see all trajectory hypotheses, not just the most confident one. This function returns the full mode distribution:
```python
def predict_all_modes(data, model_multi, cfg, device):
    """
    Runs inference and returns all M trajectory modes with their probabilities.
    
    Useful for visualization where we want to show the full distribution
    of possible futures, with line opacity proportional to confidence.
    
    Args:
        data: raw sample from L5Kit dataset
        model_multi: trained RasterTrajectoryModel
        cfg: model configuration
        device: torch device
    
    Returns:
        trajs: (M, T, 2) numpy array of trajectory coordinates
        probs: (M,) numpy array of probabilities summing to 1.0
    """
    batch = build_raster_batch_from_sample(data, cfg, device)
    
    with torch.no_grad():
        out = model_multi(batch)
        trajs = out["trajectories"][0]    # (M, T, 2)
        confs = out["confidences"][0]     # (M,) - already softmaxed
        
        probs = confs.cpu().numpy()
        trajs = trajs.cpu().numpy()
        
    return trajs, probs


def predict_best_mode(data, model_multi, cfg, device):
    """
    Returns only the highest-confidence trajectory prediction.
    
    Simpler interface when you just need the model's "best guess"
    without the full mode distribution.
    """
    trajs, probs = predict_all_modes(data, model_multi, cfg, device)
    best_idx = probs.argmax()
    return trajs[best_idx]  # (T, 2)
```

### Trajectory Distance Trimming

For cleaner visualizations, especially in animations, I trim trajectories to a maximum travel distance rather than showing the full 5-second prediction:
```python
def trim_to_distance(traj, max_dist_m):
    """
    Trims a trajectory so it doesn't extend beyond max_dist_m from the start.
    
    Useful for visualization: a 5-second prediction at highway speeds can
    extend 150+ meters, which may go off-screen. Trimming to 30m keeps
    the visualization focused on the near-term prediction.
    
    Args:
        traj: (T, 2) trajectory coordinates
        max_dist_m: maximum cumulative distance in meters
    
    Returns:
        trimmed trajectory, shape (K, 2) where K <= T
    """
    if len(traj) <= 1:
        return traj
    
    # Compute cumulative distance traveled
    displacements = traj[1:] - traj[:-1]                    # (T-1, 2)
    step_distances = np.linalg.norm(displacements, axis=1)  # (T-1,)
    cumulative_dist = np.cumsum(step_distances)             # (T-1,)
    
    # Find where we exceed the threshold
    within_range = cumulative_dist <= max_dist_m
    
    if within_range.any():
        k = within_range.sum() + 1  # +1 to include the starting point
    else:
        k = 1  # Keep at least the first point
    
    return traj[:k]
```

## Qualitative Visualizations

### ResNet vs Multimodal Comparison

![Compare Placeholder]({{ '/assets/images/team36/compare_baseline_vs_multi.png' | relative_url }})
*Fig 7. Left: ResNet produces a single trajectory that may "average" between modes. Right: Multimodal model produces diverse hypotheses capturing left turn, right turn, and straight possibilities.*

### High-Uncertainty Intersection Example

![Uncertainty Placeholder]({{ '/assets/images/team36/high_uncertainty_case_example.png' | relative_url }})
*Fig 8. At intersections, the model should predict multiple plausible futures (turn left, go straight, turn right) with appropriate confidence distribution.*

### High-Uncertainty Intersection from Model

![Uncertainty Placeholder]({{ '/assets/images/team36/high_uncertainty_case.png' | relative_url }})
*Fig 9. At intersections, this is what my model predicts with confidence distribution.*

### Animated Trajectory

This GIF shows the comparison between the ResNet and Multi-modal models. I removed all of the other 5 trajectory lines for the multi-modal model to showcase the trajectory prediction with the highest accuracy. You can also see that in the ResNet model, it's having trouble to predict the curve since really this model focuses on going straight.

![GIF Placeholder]({{ '/assets/images/team36/scene.gif' | relative_url }})
*Fig 10. Animated visualization showing predicted trajectories unfolding alongside a curved road.*

---

## Discussion

### Lessons learned

Bringing the raster into the model changed everything. Once the network could actually “see” lane geometry, intersections, and road constraints, the predictions immediately tightened up. Instead of guessing a curve based only on past positions, the model started following the shape of the road.

Adding explicit velocity features ended up being more important than I expected. Giving the model 
$$(v_x, v_y) $$ directly kept it from overshooting turns or lagging behind when an agent was accelerating. It essentially grounded the model in the agent’s motion state instead of making it infer everything from scratch.

Multimodality solved a huge chunk of the ambiguity problem. At forks, merges, and intersections, there genuinely isn’t a single correct future. With multiple modes, the model naturally spreads probability across different options, and the confidence head learned when to stay uncertain.

Soft-WTA turned out to be the difference between “multiple modes exist” and actually using them. Hard winner-take-all collapsed everything onto one trajectory. Once I switched to the temperature-scaled version, the modes stayed diverse but still got meaningful gradients.

### Limitations

The raster is treated as frozen in time, so the model never reasons about moving agents beyond what’s captured in the single frame. That also means any “interaction modeling” is purely implicit and the CNN might pick up on nearby cars, but nothing explicitly tells the model how agents influence each other.

The fixed number of modes is also a bit inflexible. Six is sometimes too many (simple highway following) and sometimes not enough (busy urban intersections). And, like most datasets, this one has biases. The model struggles with rare edge cases like U-turns, emergency braking, simply because it hasn’t seen enough of them.

### Potential Improvements

A transformer-based architecture is probably the cleanest upgrade path. Models like Wayformer or SceneTransformer handle long-range structure better than CNN+MLP. Switching to vectorized map inputs (lane polylines, connectivity graphs) would let the model reason geometrically instead of relying on pixel-based approximations.

Another promising direction is goal conditioning. Predicting trajectories relative to plausible end goals like intersection exits. That tends to make the model’s outputs more structured and easier to interpret. Finally, there’s a lot left on the table with multi-agent prediction. Handling multiple agents jointly would go a long way toward more realistic interaction modeling. Even using a short raster “video” instead of a single frame would give the model a better sense of temporal context.

---

## Conclusion

This project highlighted how inherently uncertain real driving behavior is. Vehicles do not follow fixed patterns. They respond to road geometry, evolving goals, and subtle scene cues. As the model began to incorporate these factors, its predictions became noticeably more realistic. There is still substantial room for improvement, particularly in modeling interactions between multiple agents rather than assuming a static environment. Even so, the work help build the architectural and design decisions behind trajectory prediction and shifted the process from trial-and-error to informed development. Given the limited training time, I am happy with the current model’s performance. I hope the analysis provides useful insight for anyone exploring this domain, and I look to improve this model over time.

## Code Repository


## References

[1] L5Kit Library Documentation. [https://woven-planet.github.io/l5kit/](https://woven-planet.github.io/l5kit/)

[2] Waymo Team. "VectorNet: Predicting behavior to help the Waymo Driver make better decisions." *Waymo Blog, 2020*. [https://blog.waymo.com/2020/05/vectornet.html](https://blog.waymo.com/2020/05/vectornet.html)

[3] Woven Planet Level 5. "How to Build a Motion Prediction Model for Autonomous Vehicles." *Medium, 2020*. [https://medium.com/wovenplanetlevel5/how-to-build-a-motion-prediction-model-for-autonomous-vehicles-29f7f81f1580](https://medium.com/wovenplanetlevel5/how-to-build-a-motion-prediction-model-for-autonomous-vehicles-29f7f81f1580)

[4] kool777. "Lyft Level5 EDA Training Inference." *Kaggle Notebook*. [https://www.kaggle.com/code/kool777/lyft-level5-eda-training-inference](https://www.kaggle.com/code/kool777/lyft-level5-eda-training-inference)

---

## Acknowledgments

I would like to thank the Lyft Level-5 team for releasing the motion prediction dataset and the l5kit library. 