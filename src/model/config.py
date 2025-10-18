"""Configuration management for the meme generator"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration parameters"""
    
    # Available models
    AVAILABLE_MODELS: dict = None
    
    def __post_init__(self):
        if self.AVAILABLE_MODELS is None:
            self.AVAILABLE_MODELS = {
                "Pepe Fine-tuned (LoRA)": {
                    "base": "runwayml/stable-diffusion-v1-5",
                    "lora": "MJaheen/Pepe_The_Frog_model_v1_lora",
                    "trigger_word": "pepe_style_frog",
                    "use_lora": True,
                    "use_lcm": False
                },
                "Pepe + LCM (FAST)": {
                    "base": "runwayml/stable-diffusion-v1-5",
                    "lora": "MJaheen/Pepe_The_Frog_model_v1_lora",
                    "lcm_lora": "latent-consistency/lcm-lora-sdv1-5",
                    "trigger_word": "pepe_style_frog",
                    "use_lora": True,
                    "use_lcm": True
                },
                "Base SD 1.5": {
                    "base": "runwayml/stable-diffusion-v1-5",
                    "lora": None,
                    "trigger_word": "pepe the frog",
                    "use_lora": False,
                    "use_lcm": False
                },
                "Dreamlike Photoreal 2.0": {
                    "base": "dreamlike-art/dreamlike-photoreal-2.0",
                    "lora": None,
                    "trigger_word": "pepe the frog",
                    "use_lora": False,
                    "use_lcm": False
                },
                "Openjourney v4": {
                    "base": "prompthero/openjourney-v4",
                    "lora": None,
                    "trigger_word": "pepe the frog",
                    "use_lora": False,
                    "use_lcm": False
                },
                "Tiny SD (Fast CPU)": {
                    "base": "segmind/tiny-sd",
                    "lora": None,
                    "trigger_word": "pepe the frog",
                    "use_lora": False,
                    "use_lcm": False
                },
                "Small SD (Balanced CPU)": {
                    "base": "segmind/small-sd",
                    "lora": None,
                    "trigger_word": "pepe the frog",
                    "use_lora": False,
                    "use_lcm": False
                }
            }
    
    # Default model selection
    SELECTED_MODEL: str = "Pepe Fine-tuned (LoRA)"
    
    # Model paths (will be set based on selection)
    BASE_MODEL: str = "runwayml/stable-diffusion-v1-5"
    LORA_PATH: str = "MJaheen/Pepe_The_Frog_model_v1_lora"
    USE_LORA: bool = True
    TRIGGER_WORD: str = "pepe_style_frog"
    
    # LCM settings
    USE_LCM: bool = False
    LCM_LORA_PATH: Optional[str] = None
    
    # Default generation parameters
    DEFAULT_STEPS: int = 25  # Reduced for faster CPU inference (was 50)
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
    FORCE_CPU: bool = True  # Set to True to force CPU, False to use GPU if available
    
    # Available styles
    AVAILABLE_STYLES: tuple = (
        "default", "happy", "sad", "smug", 
        "angry", "thinking", "surprised"
    )