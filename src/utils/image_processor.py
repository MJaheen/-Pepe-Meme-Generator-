"""Image processing utilities"""

from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image post-processing"""
    
    @staticmethod
    def add_meme_text(
        image: Image.Image,
        top_text: str = "",
        bottom_text: str = "",
        font_size: int = 40,
        font_path: Optional[str] = None,
    ) -> Image.Image:
        """Add classic meme text to image"""
        
        img = image.copy()
        draw = ImageDraw.Draw(img)
        
        # Load font
        try:
            if font_path:
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.truetype("impact.ttf", font_size)
        except:
            font = ImageFont.load_default()
            logger.warning("Using default font")
        
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
    def enhance_image(
        image: Image.Image,
        sharpness: float = 1.2,
        contrast: float = 1.1,
    ) -> Image.Image:
        """Apply enhancement filters"""
        
        # Sharpen
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness)
        
        # Adjust contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
        
        return image