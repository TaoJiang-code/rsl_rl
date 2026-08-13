# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learning algorithms."""

from .amp_ppo import AMPPPO
from .bfm_dagger_ppo import BFMActorModel, BFMDAggerPPO, KitovBackwardEncoderWrapper
from .distillation import Distillation
from .dwaq_ppo import DWAQPPO
from .parkour_ppo import ParkourPPO
from .ppo import PPO
from .rgmt import RGMT, RGMTActorModel
from .sonic_lora_ppo import LoRALinear, SonicActorModel, SonicCriticModel, SonicLoRAPPO

__all__ = [
    "AMPPPO",
    "BFMDAggerPPO",
    "DWAQPPO",
    "BFMActorModel",
    "KitovBackwardEncoderWrapper",
    "ParkourPPO",
    "PPO",
    "RGMT",
    "RGMTActorModel",
    "SonicLoRAPPO",
    "SonicActorModel",
    "SonicCriticModel",
    "LoRALinear",
    "Distillation",
]
