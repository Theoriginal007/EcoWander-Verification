from PIL import Image, ImageOps
import numpy as np
from typing import Tuple

def process_image_for_model(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> np.ndarray:
    """
    Process image for model input.
    
    Args:
        image_path: Path to image file
        target_size: Target dimensions (width, height)
        normalize: Whether to normalize pixel values
        
    Returns:
        Numpy array with processed image
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if not already
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize and maintain aspect ratio
            img = ImageOps.fit(img, target_size)
            
            # Convert to array and normalize
            img_array = np.array(img, dtype=np.float32)
            if normalize:
                img_array /= 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")

def detect_pink_pixels(
    image_path: str,
    threshold: float = 0.1
) -> float:
    """
    Detect percentage of pink pixels in image.
    
    Args:
        image_path: Path to image file
        threshold: Minimum pink intensity (0-1)
        
    Returns:
        Percentage of pink pixels (0-1)
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            pixels = np.array(img)
            
            # Define pink color range in RGB
            pink_pixels = (
                (pixels[:,:,0] > 200) &  # High red
                (pixels[:,:,1] > 150) &  # Medium green
                (pixels[:,:,2] > 150)    # Medium blue
            )
            
            return np.mean(pink_pixels)
            
    except Exception as e:
        raise ValueError(f"Pink detection failed: {str(e)}")