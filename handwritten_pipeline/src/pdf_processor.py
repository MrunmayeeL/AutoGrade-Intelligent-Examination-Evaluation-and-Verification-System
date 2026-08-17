import os
import logging
from typing import List, Optional
from PIL import Image, ImageEnhance
from pdf2image import convert_from_path

logger = logging.getLogger("HandwrittenPipeline.PDFProcessor")

def pdf_to_images(
    pdf_path: str,
    output_dir: str,
    dpi: int = 300,
    poppler_path: Optional[str] = None
) -> List[str]:
    """
    Converts a multi-page PDF document into high-resolution PNG images.

    Args:
        pdf_path: Path to the input PDF file.
        output_dir: Directory where the output page images should be saved.
        dpi: Resolution (dots per inch) for rendering the PDF pages. Default is 300.
        poppler_path: Path to the poppler bin directory (required on Windows). Default is None.

    Returns:
        A list of file paths to the generated page images.
    """
    logger.info(f"Loading PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Convert PDF to list of PIL Images
        logger.info(f"Converting PDF pages to images at {dpi} DPI...")
        pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
        logger.info(f"Successfully converted {len(pages)} page(s) from PDF.")
        
        image_paths = []
        for i, page in enumerate(pages, start=1):
            image_name = f"page_{i}.png"
            image_path = os.path.join(output_dir, image_name)
            page.save(image_path, "PNG")
            image_paths.append(image_path)
            logger.debug(f"Saved PDF page {i} to {image_path}")
            
        return image_paths
    except Exception as e:
        logger.error(f"Failed to convert PDF to images: {e}", exc_info=True)
        raise

def enhance_image(
    image_path: str,
    output_path: Optional[str] = None,
    contrast_factor: float = 1.5,
    brightness_factor: float = 1.2
) -> str:
    """
    Applies contrast and brightness enhancements to an image to improve OCR readability.

    Args:
        image_path: Path to the input image file.
        output_path: Path to save the enhanced image. If None, overwrites the input image.
        contrast_factor: Contrast adjustment multiplier (> 1.0 increases contrast).
        brightness_factor: Brightness adjustment multiplier (> 1.0 increases brightness).

    Returns:
        The file path of the saved enhanced image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")
        
    if output_path is None:
        output_path = image_path
        
    logger.debug(f"Enhancing image: {os.path.basename(image_path)} (Contrast: {contrast_factor}, Brightness: {brightness_factor})")
    
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale first if it helps readability (optional, but keep RGB/Grayscale based on config)
            # For handwriting, enhancing contrast on original color/grayscale works best
            enhanced_img = img.copy()
            
            # Apply contrast enhancement
            if contrast_factor != 1.0:
                contrast = ImageEnhance.Contrast(enhanced_img)
                enhanced_img = contrast.enhance(contrast_factor)
                
            # Apply brightness enhancement
            if brightness_factor != 1.0:
                brightness = ImageEnhance.Brightness(enhanced_img)
                enhanced_img = brightness.enhance(brightness_factor)
                
            # Save the processed image
            enhanced_img.save(output_path)
            logger.debug(f"Saved enhanced image to: {output_path}")
            
        return output_path
    except Exception as e:
        logger.error(f"Failed to enhance image {image_path}: {e}", exc_info=True)
        raise
