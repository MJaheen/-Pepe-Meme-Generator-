"""Pepe Meme Generator - Core generation logic"""

from typing import Optional, List
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import streamlit as st
from PIL import Image
import logging

from .config import ModelConfig

logger = logging.getLogger(__name__)


class PepeGenerator:
    """Main generator class for creating Pepe memes"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize the generator"""
        self.config = config or ModelConfig()
        self.device = self._get_device()
        self.pipe = self._load_model()
        logger.info(f"PepeGenerator initialized on {self.device}")
    
    @staticmethod
    @st.cache_resource
    def _load_model() -> StableDiffusionPipeline:
        """Load and cache the Stable Diffusion model"""
        logger.info("Loading Stable Diffusion model...")
        
        # Determine appropriate dtype based on device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        pipe = StableDiffusionPipeline.from_pretrained(
            ModelConfig.BASE_MODEL,
            torch_dtype=torch_dtype,
            safety_checker=None,  # Disabled for meme generation - users must comply with SD license
        )
        
        # Optimize scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )
        
        # Enable memory optimizations
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        
        if device == "cuda":
            pipe = pipe.to("cuda")
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except ImportError:
                logger.warning("xformers not available, using default attention")
            except Exception as e:
                logger.warning(f"Could not enable xformers: {e}")
        else:
            logger.info("Running on CPU - memory optimizations applied")
        
        logger.info("Model loaded successfully")
        return pipe
    
    @staticmethod
    def _get_device() -> str:
        """Determine the best available device"""
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def generate(
        self,
        prompt: str,
        style: str = "default",
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        width: int = 512,
        height: int = 512,
    ) -> Image.Image:
        """Generate a single Pepe meme image"""
        
        # Apply style preset
        enhanced_prompt = self._apply_style_preset(prompt, style)
        
        # Set default negative prompt
        if negative_prompt is None:
            negative_prompt = self.config.DEFAULT_NEGATIVE_PROMPT
        
        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        logger.info(f"Generating: {enhanced_prompt[:50]}...")
        
        # Generate image (removed autocast for CPU compatibility)
        output = self.pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            width=width,
            height=height,
        )
        
        return output.images[0]
    
    def generate_batch(
        self,
        prompt: str,
        num_images: int = 4,
        **kwargs
    ) -> List[Image.Image]:
        """Generate multiple variations"""
        images = []
        for i in range(num_images):
            if 'seed' not in kwargs:
                kwargs['seed'] = torch.randint(0, 100000, (1,)).item()
            
            image = self.generate(prompt, **kwargs)
            images.append(image)
            
            if 'seed' in kwargs:
                del kwargs['seed']
        
        return images
    
    @staticmethod
    def _apply_style_preset(prompt: str, style: str) -> str:
        """Apply style-specific prompt enhancements"""
        style_modifiers = {
            "happy": "cheerful, smiling, joyful",
            "sad": "melancholic, crying, emotional",
            "smug": "confident, satisfied, smirking",
            "angry": "frustrated, mad, intense",
            "thinking": "contemplative, philosophical",
            "surprised": "shocked, amazed, wide eyes",
        }
        
        base = f"pepe the frog, {prompt}"
        
        if style in style_modifiers:
            base = f"{base}, {style_modifiers[style]}"
        
        base = f"{base}, high quality, detailed, meme art"
        
        return base