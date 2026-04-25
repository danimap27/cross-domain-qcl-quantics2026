"""
noise.py — IBM Heron r2 noise model for PennyLane default.mixed simulations.

Noise parameters are derived from IBM Heron r2 hardware calibration data.
The model applies three noise channels per gate layer:
  - AmplitudeDamping  (T1 relaxation)
  - PhaseDamping      (T2 dephasing)
  - DepolarizingChannel (gate errors)

Reference hardware parameters (Heron r2, April 2025):
  T1 = 250 µs, T2 = 150 µs
  1-qubit gate time = 32 ns, 2-qubit gate time = 68 ns
  p1q = 0.0002, p2q = 0.005, readout_error = 0.012
"""

import math
from typing import Any


# Default IBM Heron r2 calibration values
IBM_HERON_R2 = {
    "T1_us": 250.0,
    "T2_us": 150.0,
    "t1q_ns": 32.0,
    "t2q_ns": 68.0,
    "p1q": 0.0002,
    "p2q": 0.005,
    "readout_error": 0.012,
}


def _gamma_amplitude(t_gate_s: float, T1_s: float) -> float:
    """Amplitude damping probability from T1 relaxation during gate time t."""
    return 1.0 - math.exp(-t_gate_s / T1_s)


def _gamma_phase(t_gate_s: float, T1_s: float, T2_s: float) -> float:
    """
    Phase damping probability.

    For a gate of duration t, the total dephasing rate is 1/T2.
    The pure dephasing component excludes the T1 contribution:
        gamma_phi = t * (1/T_phi) where 1/T_phi = 1/T2 - 1/(2*T1)
    We return the Kraus operator parameter for PhaseDamping.
    """
    lam = t_gate_s * (1.0 / T2_s - 1.0 / (2.0 * T1_s))
    if lam < 0:
        return 0.0
    return 1.0 - math.exp(-lam)


def build_noise_operators(
    n_qubits: int = 4,
    noise_params: dict | None = None,
    channels: list[str] | None = None,
) -> list[tuple]:
    """
    Build a list of noise operator descriptors for insertion into a circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    noise_params : dict or None
        Hardware calibration parameters. Defaults to IBM_HERON_R2.
    channels : list of str or None
        Subset of channels to apply. Options: "amplitude_damping", "phase_damping",
        "depolarizing". Defaults to all three.

    Returns
    -------
    ops : list of (pennylane_fn, wires, kwargs)
        Each tuple describes one noise operator application per qubit.
    """
    import pennylane as qml

    p = noise_params if noise_params is not None else IBM_HERON_R2
    active = set(channels) if channels is not None else {"amplitude_damping", "phase_damping", "depolarizing"}

    T1 = p["T1_us"] * 1e-6
    T2 = p["T2_us"] * 1e-6
    t1q = p["t1q_ns"] * 1e-9

    ops = []

    if "amplitude_damping" in active:
        gamma_ad = _gamma_amplitude(t1q, T1)
        ops.append((qml.AmplitudeDamping, list(range(n_qubits)), {"gamma": gamma_ad}))

    if "phase_damping" in active:
        gamma_pd = _gamma_phase(t1q, T1, T2)
        if gamma_pd > 0:
            ops.append((qml.PhaseDamping, list(range(n_qubits)), {"gamma": gamma_pd}))

    if "depolarizing" in active:
        p_dep = p["p1q"]
        ops.append((qml.DepolarizingChannel, list(range(n_qubits)), {"p": p_dep}))

    return ops


def get_noise_summary(noise_params: dict | None = None) -> dict[str, float]:
    """Return computed noise rates for documentation/logging."""
    p = noise_params if noise_params is not None else IBM_HERON_R2
    T1 = p["T1_us"] * 1e-6
    T2 = p["T2_us"] * 1e-6
    t1q = p["t1q_ns"] * 1e-9
    return {
        "gamma_amplitude_damping": _gamma_amplitude(t1q, T1),
        "gamma_phase_damping": _gamma_phase(t1q, T1, T2),
        "p_depolarizing_1q": p["p1q"],
        "p_depolarizing_2q": p["p2q"],
        "readout_error": p["readout_error"],
    }
