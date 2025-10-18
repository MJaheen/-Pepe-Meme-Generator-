"""Configuration management for the Pepe meme generator.

This module defines all configuration parameters for model selection,
generation settings, and application behavior. The ModelConfig dataclass
provides a centralized configuration system with sensible defaults.

"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """
    Central configuration for model and generation parameters.
    
    This dataclass contains all settings for model selection, generation
    parameters, and optimization flags. It supports multiple models including
    fine-tuned LoRA variants and fast LCM models.
    
    Attributes:
        AVAILABLE_MODELS: Dictionary of available model configurations
        SELECTED_MODEL: Currently selected model name
        BASE_MODEL: HuggingFace ID of the base Stable Diffusion model
        LORA_PATH: Path or HuggingFace ID of LoRA weights
        USE_LORA: Whether to load and use LoRA weights
        USE_LCM: Whether to use LCM (Latent Consistency Model) for fast inference
        LCM_LORA_PATH: Path to LCM-LoRA weights
        TRIGGER_WORD: Trigger word to activate fine-tuned style
        DEFAULT_STEPS: Default number of diffusion steps
        DEFAULT_GUIDANCE: Default guidance scale (CFG)
        DEFAULT_WIDTH: Default output image width
        DEFAULT_HEIGHT: Default output image height
        DEFAULT_NEGATIVE_PROMPT: Default negative prompt for all generations
        FORCE_CPU: Force CPU mode (disable GPU)
        ENABLE_XFORMERS: Enable memory-efficient attention
    """
    
    # Available models
    AVAILABLE_MODELS: dict = None
    
    def __post_init__(self):
        """
        Initialize AVAILABLE_MODELS dictionary if not already set.
        
        This method is called automatically after __init__. It populates
        the AVAILABLE_MODELS dictionary with all supported model configurations.
        Each model can have different base models, LoRA weights, and optimization flags.
        """
        if self.AVAILABLE_MODELS is None:
            self.AVAILABLE_MODELS = {
                # Primary fine-tuned model - Best quality, trained on Pepe dataset
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
    FONT_PATH: str = "src/Fonts/impact.ttf"
    
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
    FORCE_CPU: bool = False  # Set to True to force CPU, False to use GPU if available
    
    # Available styles
    AVAILABLE_STYLES: tuple = (
        "default", "happy", "sad", "smug", 
        "angry", "thinking", "surprised"
    )