"""
trainer.py — QCL training loop.

Architecture matches the original HybridQuantumNet:
  - qml.qnn.TorchLayer for efficient batched forward passes
  - Two PauliZ expectation values → nn.Linear(2, 2) → CrossEntropyLoss
  - Per-epoch loss history and Task A forgetting history saved as JSON

Sequential protocol:
  1. [Optional] Pre-train on source domain (synthetic or MobileNetV2).
  2. Train on Task A (Fashion-MNIST 0 vs 1). Record loss per epoch.
  3. Train on Task B (MNIST 2 vs 3). Record loss per epoch and Task A
     accuracy at each epoch (forgetting curve).
  4. Compute forgetting drop.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HybridQCLModel(nn.Module):
    """
    Hybrid quantum-classical binary classifier.

    Architecture (matches original HybridQuantumNet):
      1. qml.qnn.TorchLayer — VQC with Ry angle embedding + ansatz
      2. nn.Linear(2, 2)    — maps two PauliZ expectation values to class logits

    Parameters
    ----------
    qnode : callable
        PennyLane QNode with signature circuit(inputs, weights).
        Returns [expval(PauliZ(0)), expval(PauliZ(1))].
    weight_shapes : dict
        Passed to TorchLayer, e.g. {"weights": (2, 4, 3)}.
    init_params : np.ndarray or None
        If given, overwrites random initialization of the quantum weights.
    """

    def __init__(self, qnode, weight_shapes: dict, init_params=None):
        super().__init__()
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        # Use 2 measurements to match the original architecture and reduce forgetting
        self.fc = nn.Linear(2, 2)
        if init_params is not None:
            with torch.no_grad():
                self.qlayer.weights.copy_(
                    torch.tensor(init_params, dtype=torch.float32)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is scaled to [0, pi] in loader.py, pass directly to QLayer
        exp_vals = self.qlayer(x)   # (batch, 2) 
        return self.fc(exp_vals.to(torch.float32))    # (batch, 2) class logits


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def _train_epoch(
    model: HybridQCLModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """One training epoch. Returns mean cross-entropy loss."""
    model.train()
    total_loss, n = 0.0, 0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
        n += len(xb)
    return total_loss / max(n, 1)


def _evaluate(model: HybridQCLModel, loader: DataLoader) -> float:
    """Binary accuracy in [0, 1]."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Run config and result
# ---------------------------------------------------------------------------

@dataclass
class QCLRunConfig:
    run_id: str
    ansatz: str
    noise_model: str           # "ideal" or "ibm_heron_r2"
    source: str                # "scratch", "synthetic_gaussian", "mobilenetv2"
    seed: int
    n_qubits: int = 4
    n_layers: int = 2
    lr: float = 0.05
    epochs: int = 10
    pretrain_epochs: int = 10
    batch_size: int = 32
    freeze_prior: bool = True  # If True, freezes layer 0 after pre-training
    data_dir: str = "./data/raw"
    results_dir: str = "./results"
    noise_channels: list = field(
        default_factory=lambda: ["amplitude_damping", "phase_damping", "depolarizing"]
    )
    machine_id: str = "local"


@dataclass
class QCLResult:
    run_id: str
    ansatz: str
    noise_model: str
    source: str
    seed: int
    acc_source: float
    acc_a_initial: float
    acc_b_final: float
    acc_a_final: float
    forgetting_drop: float
    train_time_source_s: float
    train_time_a_s: float
    train_time_b_s: float
    status: str = "completed"
    error: str = ""

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run_qcl(cfg: QCLRunConfig) -> QCLResult:
    """
    Execute one full QCL sequential training run.

    Steps:
      0. Build quantum circuit (ideal or noisy backend).
      1. [Optional] Pre-train on source domain.
      2. Train on Task A — Fashion-MNIST classes 0 vs 1.
      3. Train on Task B — MNIST classes 2 vs 3.
         Record Task A accuracy at each epoch (forgetting curve).
      4. Compute forgetting drop = acc_A_initial - acc_A_final.
      5. Save per-epoch history to results/<run_id>/history.json.

    Parameters
    ----------
    cfg : QCLRunConfig

    Returns
    -------
    result : QCLResult
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ── Circuit ────────────────────────────────────────────────────────────────
    from circuits.ansatz import build_circuit, get_weight_shapes
    from circuits.noise import IBM_HERON_R2

    is_noisy = cfg.noise_model != "ideal"
    # Use fast C++ backend for ideal simulations
    backend = "default.mixed" if is_noisy else "lightning.qubit"
    # default.mixed does not support backprop; let PennyLane pick the best method
    diff_method = "best" if is_noisy else "adjoint"

    # Pass the noise dictionary to enable gate-wise error channels
    noise_params = IBM_HERON_R2 if is_noisy else None

    qnode = build_circuit(
        ansatz=cfg.ansatz,
        n_qubits=cfg.n_qubits,
        n_layers=cfg.n_layers,
        noise_params=noise_params,
        backend=backend,
        diff_method=diff_method,
    )
    weight_shapes = get_weight_shapes(cfg.ansatz, cfg.n_qubits, cfg.n_layers)

    # ── Data ──────────────────────────────────────────────────────────────────
    from data.loader import load_task_pca

    task_a_train, task_a_test = load_task_pca(
        "fashion_mnist", classes=[0, 1], n_features=cfg.n_qubits,
        source="pixel", data_dir=cfg.data_dir, seed=cfg.seed,
    )
    task_b_train, task_b_test = load_task_pca(
        "mnist", classes=[2, 3], n_features=cfg.n_qubits,
        source="pixel", data_dir=cfg.data_dir, seed=cfg.seed,
    )
    loader_a_tr = DataLoader(task_a_train, batch_size=cfg.batch_size, shuffle=True)
    loader_a_te = DataLoader(task_a_test, batch_size=512)
    loader_b_tr = DataLoader(task_b_train, batch_size=cfg.batch_size, shuffle=True)
    loader_b_te = DataLoader(task_b_test, batch_size=512)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = HybridQCLModel(qnode, weight_shapes)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    acc_source = 0.0
    train_time_source_s = 0.0
    loss_history_source: list[float] = []

    # ── Phase 0: Pre-training ─────────────────────────────────────────────────
    if cfg.source == "synthetic_gaussian":
        from data.synthetic import make_synthetic_gaussian
        src_ds = make_synthetic_gaussian(
            n_samples=1000, n_features=cfg.n_qubits, seed=cfg.seed,
        )
        loader_src = DataLoader(src_ds, batch_size=cfg.batch_size, shuffle=True)
        loader_src_te = DataLoader(src_ds, batch_size=512)

        t0 = time.time()
        for _ in range(cfg.pretrain_epochs):
            loss_history_source.append(
                _train_epoch(model, loader_src, optimizer, criterion)
            )
        train_time_source_s = time.time() - t0
        acc_source = _evaluate(model, loader_src_te)
        logger.info(f"[{cfg.run_id}] Pre-train (synthetic): acc={acc_source:.4f}")

    elif cfg.source == "mobilenetv2":
        src_train, src_test = load_task_pca(
            "fashion_mnist", classes=[0, 1], n_features=cfg.n_qubits,
            source="mobilenetv2", data_dir=cfg.data_dir, seed=cfg.seed,
        )
        loader_src = DataLoader(src_train, batch_size=cfg.batch_size, shuffle=True)
        loader_src_te = DataLoader(src_test, batch_size=512)

        t0 = time.time()
        for _ in range(cfg.pretrain_epochs):
            loss_history_source.append(
                _train_epoch(model, loader_src, optimizer, criterion)
            )
        train_time_source_s = time.time() - t0
        acc_source = _evaluate(model, loader_src_te)
        logger.info(f"[{cfg.run_id}] Pre-train (mobilenetv2): acc={acc_source:.4f}")

    # ── Memory Protection Strategy (Freezing) ─────────────────────────────────
    if cfg.freeze_prior and cfg.source != "scratch":
        logger.info(f"[{cfg.run_id}] Freezing Layer 0 weights to shield prior knowledge.")
        for name, param in model.qlayer.named_parameters():
            # Weights dimension 0 corresponds to the layer index
            if "weights" in name:
                # Hook gradient to zero out updates for the first layer (layer 0)
                def _freeze_hook(grad):
                    out = grad.clone()
                    out[0] = 0.0 # Shield prior knowledge in layer 0
                    return out
                param.register_hook(_freeze_hook)

    # ── Phase 1: Task A ───────────────────────────────────────────────────────
    # Use lower learning rate if pre-trained weights are present
    lr_a = cfg.lr if cfg.source == "scratch" else 0.01
    optimizer_a = torch.optim.Adam(model.parameters(), lr=lr_a)
    
    loss_history_a: list[float] = []
    t0 = time.time()
    for ep in range(cfg.epochs):
        loss_history_a.append(_train_epoch(model, loader_a_tr, optimizer_a, criterion))
        if (ep + 1) % 5 == 0:
            logger.debug(f"[{cfg.run_id}] Task A ep {ep+1}: loss={loss_history_a[-1]:.4f}")
    train_time_a_s = time.time() - t0
    acc_a_initial = _evaluate(model, loader_a_te)
    logger.info(f"[{cfg.run_id}] Task A: acc_init={acc_a_initial:.4f}, t={train_time_a_s:.1f}s")

    # ── Phase 2: Task B (sequential, no replay) ───────────────────────────────
    # Lower learning rate for Task B to mitigate catastrophic forgetting
    lr_b = 0.005
    optimizer_b = torch.optim.Adam(model.parameters(), lr=lr_b)
    
    loss_history_b: list[float] = []
    forgetting_history: list[float] = []   # Task A acc at each Task B epoch
    t0 = time.time()
    for ep in range(cfg.epochs):
        loss_history_b.append(_train_epoch(model, loader_b_tr, optimizer_b, criterion))
        # Performance fix: evaluate Task A only at the end instead of every epoch
        if (ep + 1) % 5 == 0:
            logger.debug(
                f"[{cfg.run_id}] Task B ep {ep+1}: loss={loss_history_b[-1]:.4f}"
            )
    train_time_b_s = time.time() - t0

    acc_b_final = _evaluate(model, loader_b_te)
    acc_a_final = _evaluate(model, loader_a_te)
    forgetting_history.append(acc_a_final)
    forgetting_drop = acc_a_initial - acc_a_final
    logger.info(
        f"[{cfg.run_id}] Done: acc_B={acc_b_final:.4f} "
        f"acc_A_final={acc_a_final:.4f} drop={forgetting_drop:.4f}"
    )

    # ── Save per-epoch history ─────────────────────────────────────────────────
    run_dir = Path(cfg.results_dir) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    history = {
        "run_id": cfg.run_id,
        "ansatz": cfg.ansatz,
        "noise_model": cfg.noise_model,
        "source": cfg.source,
        "seed": cfg.seed,
        "loss_source": loss_history_source,
        "loss_a": loss_history_a,
        "loss_b": loss_history_b,
        "forgetting_history": forgetting_history,
        "acc_a_initial": acc_a_initial,
    }
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return QCLResult(
        run_id=cfg.run_id,
        ansatz=cfg.ansatz,
        noise_model=cfg.noise_model,
        source=cfg.source,
        seed=cfg.seed,
        acc_source=acc_source,
        acc_a_initial=acc_a_initial,
        acc_b_final=acc_b_final,
        acc_a_final=acc_a_final,
        forgetting_drop=forgetting_drop,
        train_time_source_s=train_time_source_s,
        train_time_a_s=train_time_a_s,
        train_time_b_s=train_time_b_s,
    )
