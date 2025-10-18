"""Image Processing Utilities for Meme Creation.

This module provides utilities for post-processing generated images:
- Adding classic meme text with outlines
- Adding signatures/watermarks
- Enhancing image quality (sharpness, contrast)

All methods are static and can be used without instantiation.
The ImageProcessor class acts as a namespace for image manipulation functions.

Author: MJaheen
License: MIT
"""

from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Static utility class for image post-processing operations.
    
    This class provides methods for enhancing generated images with meme text,
    signatures, and quality improvements. All methods are static and work with
    PIL Image objects.
    
    Methods:
        add_meme_text: Add top/bottom text in classic meme style
        add_signature: Add watermark/signature to image
        enhance_image: Apply sharpness and contrast enhancements
    """
    
    @staticmethod
    def add_meme_text(
        image: Image.Image,
        top_text: str = "",
        bottom_text: str = "",
        font_size: int = 40,
        font_path: Optional[str] = None,
    ) -> Image.Image:
        """
        Add classic Impact-font meme text with white text and black outline.
        
        Creates the traditional meme format with text at the top and/or bottom
        of the image. Text is automatically converted to uppercase and rendered
        with a thick black outline for readability on any background.
        
        Args:
            image: Input PIL Image to add text to
            top_text: Text to display at top of image (default: "")
            bottom_text: Text to display at bottom of image (default: "")
            font_size: Size of the font in points (default: 40)
            font_path: Optional path to custom font file (default: uses Impact)
        
        Returns:
            PIL Image with meme text overlay (copy of original, not modified in-place)
        
        Note:
            Falls back to default font if Impact font is not found.
            Text is centered horizontally automatically.
        """
        
        img = image.copy()
        draw = ImageDraw.Draw(img)
        
        # Load font
        try:
            if font_path:
                font = ImageFont.truetype(font_path, font_size)
                print("there")
            else:
                font = ImageFont.truetype("impact.ttf", font_size)
                print("here")
        except:
            font = ImageFont.load_default()
            logger.warning("Using default font")
            print("fair")
        
        # Add top text
        if top_text:
            ImageProcessor._draw_text_with_outline(
                draw, top_text.upper(), (img.width // 2, 30), font
            )
        
        # Add bottom text
        if bottom_text:
            ImageProcessor._draw_text_with_outline(
                draw, bottom_text.upper(), (img.width // 2, img.height - 50), font
            )
        
        return img
    
    @staticmethod
    def _draw_text_with_outline(
        draw: ImageDraw.Draw,
        text: str,
        position: Tuple[int, int],
        font: ImageFont.FreeTypeFont,
        outline_width: int = 3,
    ):
        """Draw text with black outline"""
        x, y = position
        
        # Draw outline
        for adj_x in range(-outline_width, outline_width + 1):
            for adj_y in range(-outline_width, outline_width + 1):
                draw.text(
                    (x + adj_x, y + adj_y),
                    text,
                    font=font,
                    fill="black",
                    anchor="mm"
                )
        
        # Draw main text
        draw.text(position, text, font=font, fill="white", anchor="mm")
    
    @staticmethod
    def add_signature(
        image: Image.Image,
        signature: str = "MJ",
        position: str = "bottom-right",
        font_size: int = 20,
        opacity: int = 180,
    ) -> Image.Image:
        """Add a small signature/watermark to the image
        
        Args:
            image: Input image
            signature: Text to add as signature
            position: Position of signature (bottom-right, bottom-left, top-right, top-left)
            font_size: Size of the signature font
            opacity: Opacity of the signature (0-255)
        """
        img = image.copy()
        
        # Create a transparent overlay
        overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Load font
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()
                logger.warning("Using default font for signature")
        
        # Calculate text size and position
        bbox = draw.textbbox((0, 0), signature, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        padding = 10
        
        if position == "bottom-right":
            x = img.width - text_width - padding
            y = img.height - text_height - padding
        elif position == "bottom-left":
            x = padding
            y = img.height - text_height - padding
        elif position == "top-right":
            x = img.width - text_width - padding
            y = padding
        elif position == "top-left":
            x = padding
            y = padding
        else:
            x = img.width - text_width - padding
            y = img.height - text_height - padding
        
        # Draw signature with semi-transparent background
        bg_padding = 5
        draw.rectangle(
            [x - bg_padding, y - bg_padding, 
             x + text_width + bg_padding, y + text_height + bg_padding],
            fill=(0, 0, 0, opacity // 2)
        )
        
        # Draw text
        draw.text((x, y), signature, font=font, fill=(255, 255, 255, opacity))
        
        # Convert to RGB if needed and composite
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        img = Image.alpha_composite(img, overlay)
        
        # Convert back to RGB
        if img.mode == 'RGBA':
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3])
            return rgb_img
        
        return img
    
    @staticmethod
    def enhance_image(
        image: Image.Image,
        sharpness: float = 1.2,
        contrast: float = 1.1,
    ) -> Image.Image:
        """
        Apply sharpness and contrast enhancements to improve image quality.
        
        This method applies PIL's ImageEnhance filters to make the image
        crisper and more vibrant. Useful for post-processing AI-generated
        images which can sometimes appear slightly soft.
        
        Args:
            image: Input PIL Image to enhance
            sharpness: Sharpness multiplier (default: 1.2)
                - 0.0: Blurred
                - 1.0: Original sharpness
                - 2.0: Very sharp
                Recommended range: 1.0-1.5
            contrast: Contrast multiplier (default: 1.1)
                - 0.0: Gray
                - 1.0: Original contrast
                - 2.0: High contrast
                Recommended range: 1.0-1.3
        
        Returns:
            Enhanced PIL Image (modified in-place)

        """
        
        # Sharpen
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness)
        
        # Adjust contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
        
        return image