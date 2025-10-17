"""Configuration management for the meme generator"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration parameters"""
    
    # Model paths
    #BASE_MODEL: str = "runwayml/stable-diffusion-v1-5"
    #LORA_PATH: str = "./models/pepe_lora"
    BASE_MODEL: str ="stabilityai/sdxl-turbo"
    LORA_PATH: str = "MJaheen/pepe-lora-sdxl-turbo"
    
    # Default generation parameters
    DEFAULT_STEPS: int = 50
    DEFAULT_GUIDANCE: float = 7.5
    DEFAULT_WIDTH: int = 512
    DEFAULT_HEIGHT: int = 512
    
    # Negative prompt
    DEFAULT_NEGATIVE_PROMPT: str = (
        "blurry, low quality, distorted, deformed, "
        "ugly, bad anatomy, watermark, signature"
    )
    
    # Performance
    ENABLE_ATTENTION_SLICING: bool = True
    ENABLE_VAE_SLICING: bool = True
    
    # Available styles
    AVAILABLE_STYLES: tuple = (
        "default", "happy", "sad", "smug", 
        "angry", "thinking", "surprised"
    )