# pcu.py
import math, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# =========================
# Backbones / Heads
# =========================
class MLPEncoder(nn.Module):
    def __init__(self, d_in: int, width: int = 256, depth: int = 4, out_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        dims = [d_in] + [width]*(depth-1) + [out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(inplace=True)]
            if dropout > 0: layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
        self.bn = nn.LayerNorm(out_dim)  # stabilize distances

    def forward(self, x):  # [B,D] -> [B,dz]
        z = self.net(x)
        return self.bn(z)

class ScaleHead(nn.Module):
    """PCU: predict nonnegative global perturbation scale from embedding."""
    def __init__(self, dz: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dz, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )

    def forward(self, z):
        # Scale is a magnitude, so it should be nonnegative.
        return F.softplus(self.mlp(z).squeeze(-1))

class NoiseEvalHead(nn.Module):
    """Paper-style: predict per-feature |epsilon| (size D) from embedding."""
    def __init__(self, dz: int, d_out: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dz, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, d_out)
        )
    def forward(self, z):  # [B,dz] -> [B,D]
        return torch.relu(self.net(z))

# =========================
# Utilities
# =========================
def _set_seed(seed: int, deterministic: bool = False):
    import os, random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass

def pairwise_ranking_loss(dists: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    """
    dists: [B,K].
    Enforce d_i + margin <= d_j for i < j.
    """
    B, K = dists.shape

    # [B,K,K], entry i,j = d_i + margin - d_j
    diff = dists.unsqueeze(2) + margin - dists.unsqueeze(1)

    mask = torch.triu(
        torch.ones(K, K, device=dists.device, dtype=torch.bool),
        diagonal=1
    )

    loss_terms = F.relu(diff[:, mask])
    return loss_terms.mean()

def vicreg(z: torch.Tensor, std_coeff=25.0, cov_coeff=1.0) -> torch.Tensor:
    """VICReg terms on clean embeddings: variance >=1, decorrelate dims."""
    std_z = torch.sqrt(z.var(dim=0) + 1e-4)
    std_loss = torch.mean(F.relu(1.0 - std_z))
    zc = z - z.mean(dim=0)
    cov = (zc.T @ zc) / (z.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag**2).mean()
    return std_coeff * std_loss + cov_coeff * cov_loss

def noise_ladder_sigmas(K: int, sigma_min: float, sigma_max: float, device) -> torch.Tensor:
    return torch.linspace(sigma_min, sigma_max, steps=K, device=device)

def sample_gaussian_eps(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return torch.randn_like(x) * sigma

def generate_block_noise(xb: torch.Tensor, sigma_max: float = 1.0, m: int = 3) -> torch.Tensor:
    """
    Paper-style per-sample block noise:
    - Split features into m blocks; assign each block a random σ in ((i/m)*σmax, ((i+1)/m)*σmax].
    - Shuffle per-sample feature positions to diversify positions of large/small noise.
    """
    B, D = xb.shape
    device = xb.device
    eps = torch.empty_like(xb)
    block = max(1, D // m)
    for b in range(B):
        sigmas = []
        for i in range(m):
            lo = i * (sigma_max / m)
            hi = (i + 1) * (sigma_max / m)
            sig = torch.empty(1, device=device).uniform_(lo, hi).item()
            sigmas.append(sig)
        parts = []
        start = 0
        for i in range(m):
            end = D if i == m - 1 else min(D, start + block)
            parts.append(torch.randn(end - start, device=device) * sigmas[i])
            start = end
        e = torch.cat(parts)
        perm = torch.randperm(D, device=device)
        eps[b] = e[perm]
    return eps

# =========================
# Config
# =========================
@dataclass
class PCUConfig:
    # architecture
    dz: int = 128
    enc_width: int = 256
    enc_depth: int = 4
    dropout: float = 0.0

    # PCU noise ladder
    K: int = 4
    sigma_min: float = 0.05
    sigma_max: float = 1.0

    # optimization
    epochs: int = 800
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0

    # PCU losses
    rank_margin: float = 0.15
    w_rank: float = 1.0
    w_scale: float = 0.5
    w_vicreg: float = 0.1

    # PCU scoring mix
    alpha: float = 0.6
    beta: float = 0.3
    gamma: float = 0.1
    tiny_sigma: float = 0.03
    ema_decay: float = 0.99

    # score standardization and inference stability
    standardize_score_components: bool = True
    tiny_eval_copies: int = 5
    score_stats_batch: int = 4096

    # ---- NEW: paper-style noise-eval mode ----
    loss_mode: str = "PCU"          # "PCU" or "noise_eval"
    score_mode: str = "combo"        # "combo" (PCU) or "maxhead" (paper)
    noise_eval_copies: int = 2       # number of noised copies per batch item
    sigma_max_paper: float = 1.0     # σ_max for block noise
    m_parts: int = 3                 # blocks per sample for σ partitions
    use_amp: bool = True
    deterministic: bool = False
    compile_model: bool = False
    max_steps_per_epoch: int = 64
    random_batch_training: bool = True
    score_tiny_eval_copies: int = 1
    stats_tiny_eval_copies: int = 1
    max_score_stats_samples: int = 2000
    fused_optimizer: bool = True

# =========================
# Model
# =========================
class PCU:
    """
    Two modes:
      - PCU (original): ranking + scale + VICReg; score = α*scale + β*protoDist + γ*tinyNoiseSens
      - noise_eval (paper-style): predict per-feature |ε|; score = max(head_ne(f(x)))
    """
    def __init__(self, cfg: PCUConfig):
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.enc: Optional[nn.Module] = None
        self.head: Optional[nn.Module] = None        # PCU head (global scale)
        self.head_ne: Optional[nn.Module] = None     # paper head (per-feature)
        self.prototype: Optional[torch.Tensor] = None  # EMA of clean embeddings
        self.score_stats_: Optional[dict] = None

    def _build(self, d_in: int):
        self.enc = MLPEncoder(d_in, width=self.cfg.enc_width, depth=self.cfg.enc_depth,
                              out_dim=self.cfg.dz, dropout=self.cfg.dropout).to(self.cfg.device)
        # Always build PCU head; may be unused in paper mode
        self.head = ScaleHead(self.cfg.dz).to(self.cfg.device)

        if getattr(self.cfg, "compile_model", False) and hasattr(torch, "compile"):
            self.enc = torch.compile(self.enc)
            self.head = torch.compile(self.head)

    @torch.no_grad()
    def _update_proto(self, z_clean: torch.Tensor):
        if self.prototype is None:
            self.prototype = z_clean.mean(dim=0).detach()
        else:
            self.prototype = (self.cfg.ema_decay * self.prototype
                              + (1.0 - self.cfg.ema_decay) * z_clean.mean(dim=0).detach())

    @torch.no_grad()
    def _raw_score_components(self, Xs: np.ndarray, batch: int = 4096):
        """
        Compute raw PCU score components on already standardized inputs.

        Returns:
            s1: learned scale-response score
            s2: prototype-deviation score
            s3: tiny-noise sensitivity score
        """
        self.enc.eval()
        if self.head is not None:
            self.head.eval()

        device = torch.device(self.cfg.device)
        proto = self.prototype.to(device) if self.prototype is not None else None

        s1_list, s2_list, s3_list = [], [], []
        tiny_copies = max(1, int(getattr(self.cfg, "score_tiny_eval_copies", getattr(self.cfg, "tiny_eval_copies", 1))))

        for i in range(0, Xs.shape[0], batch):
            xb = torch.from_numpy(Xs[i:i + batch]).to(device, non_blocking=True)
            z = self.enc(xb)

            s1 = self.head(z)

            if proto is not None:
                s2 = (z - proto).norm(p=2, dim=1)
            else:
                s2 = torch.zeros_like(s1)

            # Average several tiny-noise probes for a more stable inference score.
            T = tiny_copies
            B, D = xb.shape

            eps = torch.randn(T, B, D, device=device, dtype=xb.dtype) * self.cfg.tiny_sigma
            xb_rep = xb.unsqueeze(0).expand(T, B, D)
            x_noisy = (xb_rep + eps).reshape(T * B, D)

            z_noisy = self.enc(x_noisy).reshape(T, B, -1)
            s3 = (z.unsqueeze(0) - z_noisy).norm(p=2, dim=2).mean(dim=0)

            s1_list.append(s1.detach().cpu())
            s2_list.append(s2.detach().cpu())
            s3_list.append(s3.detach().cpu())

        s1 = torch.cat(s1_list).numpy()
        s2 = torch.cat(s2_list).numpy()
        s3 = torch.cat(s3_list).numpy()

        return s1, s2, s3

    def _fit_score_stats(self, Xs: np.ndarray):
        if not getattr(self.cfg, "standardize_score_components", True):
            self.score_stats_ = None
            return

        max_stats_samples = int(getattr(self.cfg, "max_score_stats_samples", 10000))

        if Xs.shape[0] > max_stats_samples:
            rng = np.random.default_rng(int(getattr(self.cfg, "seed", 0)) + 999)
            idx = rng.choice(Xs.shape[0], size=max_stats_samples, replace=False)
            Xs_stats = Xs[idx]
        else:
            Xs_stats = Xs

        old_tiny = getattr(self.cfg, "score_tiny_eval_copies", getattr(self.cfg, "tiny_eval_copies", 1))
        self.cfg.score_tiny_eval_copies = int(getattr(self.cfg, "stats_tiny_eval_copies", 1))

        s1, s2, s3 = self._raw_score_components(
            Xs_stats,
            batch=int(getattr(self.cfg, "score_stats_batch", 4096))
        )

        self.cfg.score_tiny_eval_copies = old_tiny

        means = np.array([s1.mean(), s2.mean(), s3.mean()], dtype=np.float32)
        stds = np.array([s1.std(), s2.std(), s3.std()], dtype=np.float32)
        stds = np.maximum(stds, 1e-6)

        self.score_stats_ = {
            "mean": means,
            "std": stds,
        }

    def _standardize_components(self, s1: np.ndarray, s2: np.ndarray, s3: np.ndarray):
        """
        Standardize score components using nominal-training statistics.
        """
        if self.score_stats_ is None:
            return s1, s2, s3

        means = self.score_stats_["mean"]
        stds = self.score_stats_["std"]

        s1 = (s1 - means[0]) / stds[0]
        s2 = (s2 - means[1]) / stds[1]
        s3 = (s3 - means[2]) / stds[2]

        return s1, s2, s3

    def fit(self, X_norm: np.ndarray, verbose: bool = True):
        # strict determinism
        _set_seed(int(getattr(self.cfg, "seed", 0)), deterministic=getattr(self.cfg, "deterministic", False))

        # standardize on normals
        X = X_norm.astype(np.float32)
        Xs = self.scaler.fit_transform(X).astype(np.float32)
        N, D = Xs.shape
        self._build(D)

        device = torch.device(self.cfg.device)

        # Paper-style head needs D
        if self.cfg.loss_mode == "noise_eval":
            self.head_ne = NoiseEvalHead(dz=self.cfg.dz, d_out=D, hidden=max(128, self.cfg.enc_width//2)).to(device)

        # train mode
        self.enc.train()
        if self.head is not None: self.head.train()
        if self.head_ne is not None: self.head_ne.train()

        # optimizer (AMSGrad often helps on tiny datasets)
        params = list(self.enc.parameters())
        if self.head is not None: params += list(self.head.parameters())
        if self.head_ne is not None: params += list(self.head_ne.parameters())
        use_fused = (
                bool(getattr(self.cfg, "fused_optimizer", True))
                and torch.device(self.cfg.device).type == "cuda"
        )

        try:
            opt = torch.optim.AdamW(
                params,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                amsgrad=False,
                fused=use_fused,
            )
        except TypeError:
            opt = torch.optim.AdamW(
                params,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                amsgrad=False,
            )

        # per-epoch cosine schedule
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.epochs, eta_min=1e-5)

        # Keep the standardized training data on device.
        # This avoids repeated CPU -> GPU transfers at every epoch.
        X_tensor = torch.from_numpy(Xs).to(device)

        effective_batch_size = min(int(self.cfg.batch_size), max(1, N))
        steps_per_epoch = math.ceil(N / effective_batch_size)

        use_amp = bool(getattr(self.cfg, "use_amp", True)) and device.type == "cuda"
        amp_dtype = torch.float16
        scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

        use_amp = bool(getattr(self.cfg, "use_amp", True)) and device.type == "cuda"
        scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

        random_batch_training = bool(getattr(self.cfg, "random_batch_training", True))
        max_steps_per_epoch = int(getattr(self.cfg, "max_steps_per_epoch", 64))

        for ep in range(1, self.cfg.epochs + 1):

            did_optimizer_step = False
            if random_batch_training:
                steps_per_epoch = min(
                    max_steps_per_epoch,
                    max(1, math.ceil(N / effective_batch_size))
                )
            else:
                perm = torch.randperm(N, device=device)
                steps_per_epoch = math.ceil(N / effective_batch_size)

            for step in range(steps_per_epoch):

                if random_batch_training:
                    idx = torch.randint(
                        low=0,
                        high=N,
                        size=(effective_batch_size,),
                        device=device,
                    )
                else:
                    idx = perm[step * effective_batch_size: (step + 1) * effective_batch_size]
                    if idx.numel() == 0:
                        continue

                xb = X_tensor[idx]

                opt.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    if self.cfg.loss_mode == "noise_eval":
                        zc = self.enc(xb)
                        pred_clean = self.head_ne(zc)
                        loss_clean = F.mse_loss(pred_clean, torch.zeros_like(pred_clean))

                        loss_noised = 0.0
                        for _ in range(int(self.cfg.noise_eval_copies)):
                            eps = generate_block_noise(
                                xb,
                                sigma_max=self.cfg.sigma_max_paper,
                                m=self.cfg.m_parts,
                            )
                            x_hat = xb + eps
                            z_hat = self.enc(x_hat)
                            pred_eps = self.head_ne(z_hat)
                            loss_noised = loss_noised + F.mse_loss(pred_eps, eps.abs())

                        loss_noised = loss_noised / float(self.cfg.noise_eval_copies)
                        loss = loss_clean + loss_noised

                    else:
                        B, D = xb.shape
                        K = int(self.cfg.K)

                        sigmas = noise_ladder_sigmas(
                            K,
                            self.cfg.sigma_min,
                            self.cfg.sigma_max,
                            device,
                        )

                        eps = torch.randn(
                            K,
                            B,
                            D,
                            device=device,
                            dtype=xb.dtype,
                        ) * sigmas.view(K, 1, 1)

                        xk = xb.unsqueeze(0) + eps
                        xk_flat = xk.reshape(K * B, D)

                        # One encoder call only: clean + all noisy samples.
                        all_x = torch.cat([xb, xk_flat], dim=0)
                        all_z = self.enc(all_x)

                        zc = all_z[:B]
                        zk_flat = all_z[B:]
                        zk = zk_flat.reshape(K, B, -1)

                        dists = (zc.unsqueeze(0) - zk).norm(p=2, dim=2).transpose(0, 1)

                        s_target = eps.norm(p=2, dim=2) / math.sqrt(D)
                        s_target_flat = s_target.reshape(K * B)
                        s_pred_flat = self.head(zk_flat)

                        loss_rank = pairwise_ranking_loss(dists, margin=self.cfg.rank_margin)
                        loss_scale = F.smooth_l1_loss(s_pred_flat, s_target_flat)
                        loss_vr = vicreg(zc)

                        loss = (
                                self.cfg.w_rank * loss_rank
                                + self.cfg.w_scale * loss_scale
                                + self.cfg.w_vicreg * loss_vr
                        )

                if use_amp:
                    old_scale = scaler_amp.get_scale()

                    scaler_amp.scale(loss).backward()
                    scaler_amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, 5.0)
                    scaler_amp.step(opt)
                    scaler_amp.update()

                    new_scale = scaler_amp.get_scale()

                    # If the scale did not decrease, AMP did not skip the optimizer step.
                    if new_scale >= old_scale:
                        did_optimizer_step = True
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 5.0)
                    opt.step()
                    did_optimizer_step = True

                self._update_proto(zc.detach())

            if did_optimizer_step:
                sched.step()

        # fallback prototype if missing (shouldn't happen)
        if self.prototype is None:
            with torch.no_grad():
                Xb = torch.from_numpy(Xs[:min(2048, N)]).to(device)
                self.prototype = self.enc(Xb).mean(dim=0).detach()
        # Fit training-set statistics for score-component standardization.
        if self.cfg.loss_mode == "PCU" and self.cfg.score_mode == "combo":
            self._fit_score_stats(Xs)
        return self

    @torch.no_grad()
    def score_samples(self, X: np.ndarray, batch: int = 4096) -> np.ndarray:
        self.enc.eval()
        if self.head is not None:
            self.head.eval()
        if self.head_ne is not None:
            self.head_ne.eval()

        Xs = self.scaler.transform(X.astype(np.float32)).astype(np.float32)
        device = torch.device(self.cfg.device)

        # Paper-style scoring
        if self.cfg.loss_mode == "noise_eval" and self.cfg.score_mode == "maxhead":
            scores = []
            with torch.inference_mode():
                for i in range(0, Xs.shape[0], batch):
                    xb = torch.from_numpy(Xs[i:i + batch]).to(device, non_blocking=True)
                    z = self.enc(xb)
                    s = self.head_ne(z).max(dim=1).values
                    scores.append(s.detach().cpu())
            return torch.cat(scores).numpy()

        # PCU combo scoring
        s1, s2, s3 = self._raw_score_components(Xs, batch=batch)
        s1, s2, s3 = self._standardize_components(s1, s2, s3)

        scores = (
                self.cfg.alpha * s1
                + self.cfg.beta * s2
                + self.cfg.gamma * s3
        )

        return scores.astype(np.float32)

    @torch.no_grad()
    def predict(self, X: np.ndarray, tau: Optional[float] = None) -> np.ndarray:
        scores = self.score_samples(X)
        if tau is None:
            tau = float(np.quantile(scores, 0.95))
        return (scores > tau).astype(np.int32)
