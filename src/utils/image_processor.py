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
        """Apply enhancement filters"""
        
        # Sharpen
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness)
        
        # Adjust contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
        
        return image