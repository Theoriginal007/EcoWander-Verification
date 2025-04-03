import hashlib
from PIL import Image, ImageFilter
import numpy as np
from typing import Dict

def generate_image_hash(image_path: str, hash_size: int = 16) -> str:
    """
    Generate perceptual hash for image.
    
    Args:
        image_path: Path to image file
        hash_size: Size of hash to generate
        
    Returns:
        Hexadecimal hash string
    """
    try:
        # Open and process image
        with Image.open(image_path) as img:
            # Convert to grayscale and resize
            img = img.convert('L').resize(
                (hash_size, hash_size), 
                Image.LANCZOS
            )
            
            # Calculate average pixel value
            pixels = np.array(img)
            avg = pixels.mean()
            
            # Generate hash bits (1 if pixel > avg, else 0)
            bits = ((pixels > avg).flatten() * 1).tolist()
            
            # Convert bits to hexadecimal
            hash_hex = ''.join([
                f'{int("".join(map(str, bits[i:i+4])), 2):x}'
                for i in range(0, len(bits), 4)
            ])
            
            return hash_hex
            
    except Exception as e:
        raise ValueError(f"Hash generation failed: {str(e)}")

def check_image_manipulation(image_path: str) -> Dict:
    """
    Check for signs of image manipulation.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with manipulation detection results
    """
    try:
        with Image.open(image_path) as img:
            # Check basic manipulation indicators
            results = {
                "has_transparency": img.mode in ('RGBA', 'LA'),
                "has_alpha": 'transparency' in img.info,
                "has_thumbnails": 'thumbnail' in img.info,
                "has_editing_software_tags": False,  # Would check EXIF in real impl
                "is_edited": False
            }
            
            # Simple edge detection analysis
            edges = img.filter(ImageFilter.FIND_EDGES())
            edge_var = np.array(edges).var()
            results["edge_variance"] = edge_var
            results["is_edited"] = edge_var > 500  # Arbitrary threshold
            
            return results
            
    except Exception as e:
        return {
            "error": str(e),
            "is_edited": True  # Assume edited if we can't check
        }