import exifread
from geopy.distance import geodesic
from typing import Tuple, Optional, Dict, List

def get_image_location(image_path: str) -> Optional[Tuple[float, float]]:
    """
    Extract GPS coordinates from image EXIF data.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Tuple of (latitude, longitude) or None if no EXIF data
    """
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            
            if not all(k in tags for k in ['GPS GPSLatitude', 'GPS GPSLongitude']):
                return None
                
            lat = _convert_to_degrees(tags['GPS GPSLatitude'].values)
            lon = _convert_to_degrees(tags['GPS GPSLongitude'].values)
            
            if tags['GPS GPSLatitudeRef'].values != 'N':
                lat = -lat
            if tags['GPS GPSLongitudeRef'].values != 'E':
                lon = -lon
                
            return (lat, lon)
            
    except Exception:
        return None

def get_nearest_eco_location(
    coordinates: Tuple[float, float],
    eco_locations: List[Dict]
) -> Tuple[Dict, float]:
    """
    Find nearest eco-location to given coordinates.
    
    Args:
        coordinates: Tuple of (lat, lng)
        eco_locations: List of eco-location dictionaries
        
    Returns:
        Tuple of (nearest_location, distance_in_meters)
    """
    nearest = None
    min_distance = float('inf')
    
    for loc in eco_locations:
        distance = geodesic(coordinates, loc['coordinates']).meters
        if distance < min_distance:
            min_distance = distance
            nearest = loc
            
    return nearest, min_distance

def _convert_to_degrees(value) -> float:
    """Convert EXIF GPS coordinates to decimal degrees."""
    d, m, s = value
    return float(d) + (float(m) / 60.0) + (float(s) / 3600.0)