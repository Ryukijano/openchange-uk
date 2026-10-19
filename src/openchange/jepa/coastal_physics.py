"""Object-oriented coastal hydrodynamic and morphodynamic forcing models.

Implements the physical nearshore formulations from OpenEarth, Stockdon et al.
(2006), and ShoreFor (Yates et al. 2009; Splinter et al. 2014) to separate
reversible instantaneous waterline shifts (tide, atmospheric storm surge, wave
setup/runup) from persistent morphological shoreline evolution driven by
cumulative wave-energy disequilibrium.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor

# Earth-Observation / Coastal history-stratum boundaries in seconds: (0d, 1d, 7d, 30d, 90d)
EO_HISTORY_STRATA_SECONDS: tuple[float, ...] = (
    0.0,
    86400.0,
    604800.0,
    2592000.0,
    7776000.0,
)


@dataclass(frozen=True)
class CoastalHydrodynamicConfig:
    """Physical constants and beach parameters for nearshore forcing."""

    beach_slope: float = 0.1
    fall_velocity: float = 0.04
    memory_decay: float = 0.85
    gravity: float = 9.80665
    seawater_density: float = 1025.0

    def __post_init__(self) -> None:
        if self.beach_slope <= 0.0 or not math.isfinite(self.beach_slope):
            raise ValueError("beach_slope must be finite and positive")
        if self.fall_velocity <= 0.0 or not math.isfinite(self.fall_velocity):
            raise ValueError("fall_velocity must be finite and positive")
        if not (0.0 < self.memory_decay <= 1.0) or not math.isfinite(self.memory_decay):
            raise ValueError("memory_decay must be in (0, 1]")
        if self.gravity <= 0.0 or self.seawater_density <= 0.0:
            raise ValueError("gravity and seawater_density must be positive")


@dataclass(frozen=True)
class StockdonRunupResult:
    """Typed output of the Stockdon et al. (2006) wave setup and runup model."""

    wavelength_l0: Tensor
    iribarren_xi0: Tensor
    wave_setup: Tensor
    swash_incident: Tensor
    swash_infragravity: Tensor
    r2: Tensor
    r2_dissipative: Tensor
    wave_power_kw_m: Tensor
    total_water_level: Tensor
    hydro_excursion: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        return {
            "wavelength_l0": self.wavelength_l0,
            "iribarren_xi0": self.iribarren_xi0,
            "wave_setup": self.wave_setup,
            "swash_incident": self.swash_incident,
            "swash_infragravity": self.swash_infragravity,
            "r2": self.r2,
            "r2_dissipative": self.r2_dissipative,
            "wave_power_kw_m": self.wave_power_kw_m,
            "total_water_level": self.total_water_level,
            "hydro_excursion": self.hydro_excursion,
        }

    def __getitem__(self, key: str) -> Tensor:
        return self.as_dict()[key]


@dataclass(frozen=True)
class ShoreForDisequilibriumResult:
    """Typed output of the ShoreFor (Yates et al. 2009) wave-power disequilibrium model."""

    dean_omega: Tensor
    omega_equilibrium: Tensor
    wave_power_kw_m: Tensor
    disequilibrium: Tensor
    cumulative_erosion_stress: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        return {
            "dean_omega": self.dean_omega,
            "omega_equilibrium": self.omega_equilibrium,
            "wave_power_kw_m": self.wave_power_kw_m,
            "disequilibrium": self.disequilibrium,
            "cumulative_erosion_stress": self.cumulative_erosion_stress,
        }

    def __getitem__(self, key: str) -> Tensor:
        return self.as_dict()[key]


class CoastalHydrodynamicModel(nn.Module):
    """Object-oriented coastal hydrodynamic and morphodynamic forcing module.

    Encapsulates Stockdon et al. (2006) wave setup/runup and ShoreFor causal
    disequilibrium stress to construct physically grounded forcing and baseline
    tensors ``[B, T, 8]`` for the spatiotemporal JEPA world model.
    """

    NUM_FORCING_CHANNELS: int = 8

    def __init__(self, config: CoastalHydrodynamicConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else CoastalHydrodynamicConfig()

    def compute_stockdon(
        self,
        hs: Tensor,
        tp: Tensor,
        *,
        beach_slope: Tensor | float | None = None,
        tide: Tensor | None = None,
        surge: Tensor | None = None,
    ) -> StockdonRunupResult:
        """Compute Stockdon et al. (2006) setup, 2% runup R_2, wave power, and excursion."""
        if hs.shape != tp.shape or hs.numel() == 0:
            raise ValueError("hs and tp must share identical non-empty shape")
        if not (torch.isfinite(hs).all() and torch.isfinite(tp).all()):
            raise ValueError("hs and tp must be finite")
        if bool((hs < 0.0).any().item()) or bool((tp <= 0.0).any().item()):
            raise ValueError("require hs >= 0 and tp > 0")

        raw_slope = self.config.beach_slope if beach_slope is None else beach_slope
        slope = (
            torch.full_like(hs, float(raw_slope))
            if isinstance(raw_slope, (int, float))
            else raw_slope.to(device=hs.device, dtype=hs.dtype)
        )
        if (
            slope.shape != hs.shape
            or not torch.isfinite(slope).all()
            or bool((slope <= 0.0).any().item())
        ):
            raise ValueError("beach_slope must be positive and match hs shape")

        eta_tide = (
            torch.zeros_like(hs) if tide is None else tide.to(device=hs.device, dtype=hs.dtype)
        )
        eta_surge = (
            torch.zeros_like(hs) if surge is None else surge.to(device=hs.device, dtype=hs.dtype)
        )
        if eta_tide.shape != hs.shape or eta_surge.shape != hs.shape:
            raise ValueError("tide and surge must match hs shape")

        g = float(self.config.gravity)
        rho = float(self.config.seawater_density)

        l0 = (g / (2.0 * math.pi)) * tp.square()
        sqrt_hl0 = (hs.clamp_min(0.0) * l0).sqrt()
        tan_beta = torch.tan(slope) if bool((slope < 0.5).all().item()) else slope
        xi0 = slope / (hs.clamp_min(1e-8) / l0).sqrt()

        wave_setup = 0.35 * slope * sqrt_hl0
        swash_incident = 0.75 * slope * sqrt_hl0
        swash_infragravity = 0.06 * sqrt_hl0
        swash_total = (swash_incident.square() + swash_infragravity.square()).sqrt()
        r2_general = 1.1 * (wave_setup + 0.5 * swash_total)
        r2_dissipative = 0.043 * sqrt_hl0

        wave_power_kw_m = (rho * (g**2) / (64.0 * math.pi)) * hs.square() * tp * 1e-3
        total_water_level = eta_tide + eta_surge + wave_setup
        hydro_excursion = total_water_level / tan_beta.clamp_min(1e-4)

        return StockdonRunupResult(
            wavelength_l0=l0,
            iribarren_xi0=xi0,
            wave_setup=wave_setup,
            swash_incident=swash_incident,
            swash_infragravity=swash_infragravity,
            r2=r2_general,
            r2_dissipative=r2_dissipative,
            wave_power_kw_m=wave_power_kw_m,
            total_water_level=total_water_level,
            hydro_excursion=hydro_excursion,
        )

    def compute_shorefor(self, hs: Tensor, tp: Tensor) -> ShoreForDisequilibriumResult:
        """Compute ShoreFor causal wave-power disequilibrium and erosion stress on [B, T]."""
        if hs.ndim != 2 or tp.shape != hs.shape or hs.numel() == 0:
            raise ValueError("hs and tp must share 2D shape [B, T]")
        if bool((hs < 0.0).any().item()) or bool((tp <= 0.0).any().item()):
            raise ValueError("require hs >= 0 and tp > 0")

        ws = float(self.config.fall_velocity)
        gamma = float(self.config.memory_decay)
        g = float(self.config.gravity)
        rho = float(self.config.seawater_density)

        omega = hs / (ws * tp)
        power = (rho * (g**2) / (64.0 * math.pi)) * hs.square() * tp * 1e-3
        power_sqrt = power.clamp_min(0.0).sqrt()

        omega_eq_steps: list[Tensor] = []
        erosion_stress_steps: list[Tensor] = []
        num_acc = torch.zeros_like(hs[:, 0])
        den_acc = torch.zeros_like(hs[:, 0])
        stress_acc = torch.zeros_like(hs[:, 0])

        for t in range(hs.shape[1]):
            p_t = power[:, t].clamp_min(1e-6)
            num_acc = gamma * num_acc + p_t * omega[:, t]
            den_acc = gamma * den_acc + p_t
            omega_eq_t = num_acc / den_acc.clamp_min(1e-8)
            omega_eq_steps.append(omega_eq_t)

            diseq_t = power_sqrt[:, t] * (omega[:, t] - omega_eq_t)
            stress_acc = gamma * stress_acc + functional.relu(diseq_t)
            erosion_stress_steps.append(stress_acc)

        omega_eq = torch.stack(omega_eq_steps, dim=1)
        disequilibrium = power_sqrt * (omega - omega_eq)
        erosion_stress = torch.stack(erosion_stress_steps, dim=1)

        return ShoreForDisequilibriumResult(
            dean_omega=omega,
            omega_equilibrium=omega_eq,
            wave_power_kw_m=power,
            disequilibrium=disequilibrium,
            cumulative_erosion_stress=erosion_stress,
        )

    def build_forcing_and_baseline(
        self,
        hs: Tensor,
        tp: Tensor,
        *,
        tide: Tensor | None = None,
        surge: Tensor | None = None,
        beach_slope: Tensor | float | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Build 8-channel coastal hydrodynamic forcing and equilibrium baseline tensors [B, T, 8]."""
        stock = self.compute_stockdon(
            hs, tp, beach_slope=beach_slope, tide=tide, surge=surge
        )
        shore = self.compute_shorefor(hs, tp)
        power_sqrt = shore.wave_power_kw_m.clamp_min(0.0).sqrt()

        forcing = torch.stack(
            (
                hs,
                tp,
                stock.wave_setup,
                stock.r2,
                power_sqrt,
                shore.disequilibrium,
                stock.total_water_level,
                stock.hydro_excursion,
            ),
            dim=-1,
        )

        hs_eq = shore.omega_equilibrium * float(self.config.fall_velocity) * tp
        stock_eq = self.compute_stockdon(hs_eq, tp, beach_slope=beach_slope)
        power_eq_sqrt = stock_eq.wave_power_kw_m.clamp_min(0.0).sqrt()
        zeros = torch.zeros_like(hs)
        baseline = torch.stack(
            (
                hs_eq,
                tp,
                stock_eq.wave_setup,
                stock_eq.r2,
                power_eq_sqrt,
                zeros,
                stock_eq.wave_setup,
                stock_eq.hydro_excursion,
            ),
            dim=-1,
        )
        return forcing, baseline

    def forward(
        self,
        hs: Tensor,
        tp: Tensor,
        *,
        tide: Tensor | None = None,
        surge: Tensor | None = None,
        beach_slope: Tensor | float | None = None,
    ) -> tuple[Tensor, Tensor]:
        return self.build_forcing_and_baseline(
            hs, tp, tide=tide, surge=surge, beach_slope=beach_slope
        )


def stockdon_wave_setup_and_runup(
    hs: Tensor,
    tp: Tensor,
    beach_slope: Tensor | float = 0.1,
    *,
    tide: Tensor | None = None,
    surge: Tensor | None = None,
    g: float = 9.80665,
    rho: float = 1025.0,
) -> dict[str, Tensor]:
    """Functional wrapper around ``CoastalHydrodynamicModel.compute_stockdon``."""
    slope_scalar = float(beach_slope) if isinstance(beach_slope, (int, float)) else 0.1
    cfg = CoastalHydrodynamicConfig(
        beach_slope=slope_scalar, gravity=g, seawater_density=rho
    )
    return (
        CoastalHydrodynamicModel(cfg)
        .compute_stockdon(hs, tp, beach_slope=beach_slope, tide=tide, surge=surge)
        .as_dict()
    )


def shorefor_disequilibrium_stress(
    hs: Tensor,
    tp: Tensor,
    *,
    fall_velocity: float = 0.04,
    memory_decay: float = 0.85,
    g: float = 9.80665,
    rho: float = 1025.0,
) -> dict[str, Tensor]:
    """Functional wrapper around ``CoastalHydrodynamicModel.compute_shorefor``."""
    cfg = CoastalHydrodynamicConfig(
        fall_velocity=fall_velocity,
        memory_decay=memory_decay,
        gravity=g,
        seawater_density=rho,
    )
    return CoastalHydrodynamicModel(cfg).compute_shorefor(hs, tp).as_dict()


def build_coastal_forcing_tensor(
    hs: Tensor,
    tp: Tensor,
    *,
    tide: Tensor | None = None,
    surge: Tensor | None = None,
    beach_slope: Tensor | float = 0.1,
    fall_velocity: float = 0.04,
    memory_decay: float = 0.85,
) -> tuple[Tensor, Tensor]:
    """Functional wrapper around ``CoastalHydrodynamicModel.build_forcing_and_baseline``."""
    slope_scalar = float(beach_slope) if isinstance(beach_slope, (int, float)) else 0.1
    cfg = CoastalHydrodynamicConfig(
        beach_slope=slope_scalar,
        fall_velocity=fall_velocity,
        memory_decay=memory_decay,
    )
    return CoastalHydrodynamicModel(cfg).build_forcing_and_baseline(
        hs, tp, tide=tide, surge=surge, beach_slope=beach_slope
    )


__all__ = [
    "CoastalHydrodynamicConfig",
    "CoastalHydrodynamicModel",
    "EO_HISTORY_STRATA_SECONDS",
    "ShoreForDisequilibriumResult",
    "StockdonRunupResult",
    "build_coastal_forcing_tensor",
    "shorefor_disequilibrium_stress",
    "stockdon_wave_setup_and_runup",
]
