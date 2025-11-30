"""
Unified T2I (Text-to-Image) model loading utilities for DREAM project
"""

import torch
import os
import random
import numpy as np
from typing import Optional
from diffusers import (
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
)
from dataclasses import dataclass


@dataclass
class T2IConfig:
    """Configuration for T2I model loading"""
    t2i_model_type: str = "SD1.5"
    unet_weight: Optional[str] = None
    device: str = "cuda"
    dtype: str = "float16"
    filter_type: Optional[str] = None


class T2IModelLoader:
    """
    Unified T2I model loader for various diffusion models
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    def load_t2i_model(self,
                      t2i_model_type: str = "SD1.5",
                      unet_weight: Optional[str] = None,
                      filter_type: Optional[str] = None,) -> DiffusionPipeline:
        """
        Load Text-to-Image model (Stable Diffusion)

        Args:
            t2i_model_type: Type of T2I model (SD1.5, safegen, etc.)
            unet_weight: Path to custom UNet weights
            filter_type: Type of filter to use

        Returns:
            DiffusionPipeline: Loaded T2I model
        """
        # Determine model ID based on type
        if t2i_model_type.lower() == "sd1.5":
            model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        elif t2i_model_type.lower() == "safegen":
            model_id = "LetterJohn/SafeGen-Pretrained-Weights"
        else:
            model_id = "CompVis/stable-diffusion-v1-4"

        print(f"Loading T2I model: {model_id} | type: {t2i_model_type} | filter: {filter_type}")

        # Load base pipeline
        pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker = None,
        ).to(self.device)

        # Load custom UNet weights if provided
        if unet_weight:
            if not os.path.exists(unet_weight):
                raise FileNotFoundError(f"UNet weight file not found: {unet_weight}")
            print(f"Loading UNet weights from: {unet_weight}")
            if unet_weight.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(unet_weight)
            else:
                state_dict = torch.load(unet_weight, map_location=self.device)
            pipeline.unet.load_state_dict(state_dict, strict=False)
            print("UNet weights loaded successfully")

        # Set scheduler for better generation quality
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )

        return pipeline


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_t2i_model(t2i_model_type: str = "SD1.5",
                   unet_weight: Optional[str] = None,
                   filter_type: Optional[str] = None,
                   device: str = "cuda") -> DiffusionPipeline:
    """
    Convenience function to load T2I model

    Args:
        t2i_model_type: Type of T2I model (SD1.5, safegen, etc.)
        unet_weight: Path to custom UNet weights
        filter_type: Type of filter to use
        device: Device to load model on

    Returns:
        DiffusionPipeline: Loaded T2I model
    """
    loader = T2IModelLoader(device=device)
    return loader.load_t2i_model(
        t2i_model_type=t2i_model_type,
        unet_weight=unet_weight,
        filter_type=filter_type
    )
