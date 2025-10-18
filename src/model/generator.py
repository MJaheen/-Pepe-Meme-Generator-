"""Pepe Meme Generator - Core generation logic.

This module contains the main PepeGenerator class which handles:
- Loading and caching Stable Diffusion models
- Managing LoRA and LCM-LoRA adapters
- Configuring schedulers and optimizations
- Generating images from text prompts
- Progress tracking during generation

The generator supports multiple models, automatic GPU/CPU detection,
memory optimizations, and both standard and fast (LCM) inference modes.

"""

from typing import Optional, List, Callable
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, LCMScheduler
import streamlit as st
from PIL import Image
import logging
import os

from .config import ModelConfig

logger = logging.getLogger(__name__)


class PepeGenerator:
    """
    Main generator class for creating Pepe meme images.
    
    This class manages the entire image generation pipeline including:
    - Model loading and caching (with Streamlit cache_resource)
    - LoRA and LCM-LoRA adapter management
    - Scheduler configuration (DPM Solver or LCM)
    - Memory optimizations (attention slicing, VAE slicing, xformers)
    - Device management (automatic CUDA/CPU detection)
    - Progress tracking callbacks
    
    The generator is designed to work efficiently on both GPU and CPU,
    with automatic optimizations based on available hardware.
    
    Attributes:
        config: ModelConfig instance with generation settings
        device: Torch device ('cuda' or 'cpu')
        pipe: Cached StableDiffusionPipeline instance
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the Pepe generator with configuration.
        
        Sets up the generator by determining the compute device (GPU/CPU),
        loading the model pipeline, and caching it for reuse. The model
        loading is cached using Streamlit's cache_resource decorator to avoid
        reloading on every interaction.
        
        Args:
            config: ModelConfig instance. If None, uses default configuration.
        
        Example:
            >>> config = ModelConfig()
            >>> config.USE_LCM = True  # Enable fast generation
            >>> generator = PepeGenerator(config)
        """
        self.config = config or ModelConfig()
        self.device = self._get_device(self.config.FORCE_CPU)
        self.pipe = self._load_model(
            self.config.BASE_MODEL,
            self.config.USE_LORA,
            self.config.LORA_PATH,
            self.config.FORCE_CPU,
            self.config.USE_LCM,
            self.config.LCM_LORA_PATH
        )
        logger.info(f"PepeGenerator initialized on {self.device}")

    @staticmethod
    @st.cache_resource
    def _load_model(base_model: str, use_lora: bool, lora_path: Optional[str], 
                    force_cpu: bool = False, use_lcm: bool = False, 
                    lcm_lora_path: Optional[str] = None) -> StableDiffusionPipeline:
        """Load and cache the Stable Diffusion model with LoRA and LCM support"""
        logger.info("="*60)
        logger.info("LOADING NEW MODEL PIPELINE")
        logger.info(f"Base Model: {base_model}")
        logger.info(f"LoRA Enabled: {use_lora}")
        if use_lora and lora_path:
            logger.info(f"LoRA Path: {lora_path}")
        logger.info(f"LCM Enabled: {use_lcm}")
        if use_lcm and lcm_lora_path:
            logger.info(f"LCM-LoRA Path: {lcm_lora_path}")
        logger.info(f"Force CPU: {force_cpu}")
        logger.info("="*60)

        # Determine appropriate dtype based on device
        if force_cpu:
            device = "cpu"
            logger.info("🔧 FORCED CPU MODE - GPU disabled for testing")
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        torch_dtype = torch.float16 if (device == "cuda" and not force_cpu) else torch.float32
        logger.info(f"Using device: {device}, dtype: {torch_dtype}")

        pipe = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            safety_checker=None,  # Disabled for meme generation - users must comply with SD license
        )

        # Load LoRA weights if configured
        if use_lora and lora_path:
            logger.info(f"Loading LoRA weights from: {lora_path}")
            try:
                # Check if it's a local path or Hugging Face model ID
                # Explicitly name it "pepe" to avoid "default_0" naming
                if os.path.exists(lora_path):
                    # Local path
                    pipe.load_lora_weights(lora_path, adapter_name="pepe")
                    logger.info("LoRA weights loaded successfully from local path")
                elif "/" in lora_path:
                    # Hugging Face model ID (format: username/model_name)
                    pipe.load_lora_weights(lora_path, adapter_name="pepe")
                    logger.info(f"✅ LoRA weights loaded successfully from Hugging Face: {lora_path}")
                else:
                    logger.warning(f"Invalid LoRA path format: {lora_path}")
                
                # If not using LCM, set Pepe LoRA as the active adapter
                if not use_lcm:
                    pipe.set_adapters(["pepe"])
                    logger.info("✅ Pepe LoRA active")
            except Exception as e:
                logger.error(f"Failed to load LoRA weights: {e}")
                logger.info("Continuing without LoRA weights...")
        
        # Load LCM-LoRA on top if configured (this enables fast inference!)
        if use_lcm and lcm_lora_path:
            logger.info(f"Loading LCM-LoRA from: {lcm_lora_path}")
            try:
                # Load LCM-LoRA as a separate adapter
                pipe.load_lora_weights(lcm_lora_path, adapter_name="lcm")
                logger.info("✅ LCM-LoRA loaded successfully")
                
                # If we have both Pepe LoRA and LCM-LoRA, fuse them
                if use_lora:
                    logger.info("Fusing Pepe LoRA + LCM-LoRA adapters...")
                    # Use the correct adapter names: "pepe" and "lcm"
                    pipe.set_adapters(["pepe", "lcm"], adapter_weights=[1.0, 1.0])
                    logger.info("✅ Both LoRAs fused successfully (pepe + lcm)")
                else:
                    # Only LCM, set it as active
                    pipe.set_adapters(["lcm"])
                    logger.info("✅ LCM-LoRA active (solo mode)")
            except Exception as e:
                logger.error(f"Failed to load LCM-LoRA: {e}")
                logger.info("Continuing without LCM...")
                use_lcm = False

        # Set appropriate scheduler based on LCM mode
        if use_lcm:
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            logger.info("⚡ Using LCM Scheduler (few-step mode)")
        else:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config
            )
            logger.info("🔧 Using DPM Solver Scheduler (standard mode)")

        # Enable memory optimizations
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()

        if device == "cuda" and not force_cpu:
            pipe = pipe.to("cuda")
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except ImportError:
                logger.warning("xformers not available, using default attention")
            except Exception as e:
                logger.warning(f"Could not enable xformers: {e}")
        else:
            if force_cpu:
                logger.info("Running on CPU - FORCED for testing")
            else:
                logger.info("Running on CPU - memory optimizations applied")

        logger.info("Model loaded successfully")
        return pipe

    @staticmethod
    def _get_device(force_cpu: bool = False) -> str:
        """Determine the best available device"""
        if force_cpu:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        style: str = "default",
        raw_prompt: bool = False
    ) -> Image:
        """
        Generate a Pepe meme image from a text prompt.
        
        This method runs the diffusion process to generate an image based on
        the provided text prompt. It supports various parameters to control
        the generation quality, style, and randomness.
        
        Args:
            prompt: Text description of the desired image. For best results with
                the fine-tuned model, include the trigger word 'pepe_style_frog'.
            negative_prompt: Text describing what to avoid in the image.
                If None, uses default from config.
            num_inference_steps: Number of denoising steps (4-8 for LCM, 20-50 normal).
            guidance_scale: CFG scale (1.0-2.0 for LCM, 5.0-15.0 normal).
            width: Output image width in pixels (must be divisible by 8).
            height: Output image height in pixels (must be divisible by 8).
            seed: Random seed for reproducible generation.
            progress_callback: Optional callback(current_step, total_steps).
            style: Style preset to apply ('default', 'happy', 'sad', 'smug', 'angry', 'thinking', 'surprised').
            raw_prompt: If True, use prompt as-is without trigger words or style modifiers.
        
        Returns:
            PIL Image object containing the generated image.
        """
        # Handle raw prompt mode - use prompt as-is if requested
        if raw_prompt:
            enhanced_prompt = prompt
        else:
            # Apply style preset if not in raw mode
            enhanced_prompt = self._apply_style_preset(prompt, style) if style != "default" else prompt

        # Set default negative prompt
        if negative_prompt is None:
            negative_prompt = self.config.DEFAULT_NEGATIVE_PROMPT

        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        logger.info(f"Generating: {enhanced_prompt[:50]}...")
        logger.debug(f"Full prompt: {enhanced_prompt}")
        logger.debug(f"Model config - Base: {self.config.BASE_MODEL}, LoRA: {self.config.USE_LORA}")

        # Create callback wrapper if provided (using new diffusers API)
        callback_on_step_end_fn = None
        if progress_callback:
            def callback_on_step_end_fn(pipe, step, timestep, callback_kwargs):
                progress_callback(step + 1, num_inference_steps)
                return callback_kwargs

        # Generate image (removed autocast for CPU compatibility)
        output = self.pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            width=width,
            height=height,
            callback_on_step_end=callback_on_step_end_fn,
        )

        return output.images[0]

    def generate_batch(
        self,
        prompt: str,
        num_images: int = 4,
        **kwargs
    ) -> List[Image.Image]:
        """Generate multiple variations with callback support"""
        images = []
        for i in range(num_images):
            if 'seed' not in kwargs:
                kwargs['seed'] = torch.randint(0, 100000, (1,)).item()

            image = self.generate(prompt, **kwargs)
            images.append(image)

            if 'seed' in kwargs:
                del kwargs['seed']

        return images

    def _apply_style_preset(self, prompt: str, style: str) -> str:
        """Apply style-specific prompt enhancements"""
        style_modifiers = {
            "happy": "cheerful, smiling, joyful",
            "sad": "melancholic, crying, emotional",
            "smug": "confident, satisfied, smirking",
            "angry": "frustrated, mad, intense",
            "thinking": "contemplative, philosophical",
            "surprised": "shocked, amazed, wide eyes",
        }
        
        # Use trigger word from config
        trigger_word = self.config.TRIGGER_WORD
        
        base = f"{trigger_word}, {prompt}"

        if style in style_modifiers:
            base = f"{base}, {style_modifiers[style]}"

        base = f"{base}, high quality, detailed, meme art"

        return base