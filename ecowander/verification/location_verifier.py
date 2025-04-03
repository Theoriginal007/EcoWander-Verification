from geopy.distance import geodesic
from ecowander.services.geo_utils import (
    get_image_location,
    get_nearest_eco_location
)
from ecowander.config.eco_locations import KNOWN_ECO_LOCATIONS
from typing import Tuple, Dict, Optional

class LocationVerifier:
    def __init__(self, max_distance_meters: float = 100):
        self.max_distance = max_distance_meters
        
    def verify_location(
        self,
        image_path: str,
        user_location: Tuple[float, float],
        timestamp: Optional[float] = None
    ) -> Dict:
        """
        Verify location matches known eco-spots.
        
        Args:
            image_path: Path to image with potential EXIF data
            user_location: Tuple of (lat, lng) from user
            timestamp: Optional timestamp for validation
            
        Returns:
            Dictionary with verification results
        """
        try:
            # Try to get location from image first
            img_location = get_image_location(image_path)
            actual_location = img_location or user_location
            
            if not actual_location:
                raise ValueError("No location data provided")
            
            # Find nearest known location
            nearest, distance = get_nearest_eco_location(
                actual_location,
                KNOWN_ECO_LOCATIONS
            )
            
            # Calculate verification score
            if distance <= self.max_distance:
                score = 1.0
            else:
                # Linear decay beyond max distance
                score = max(0, 1 - (distance / (self.max_distance * 10)))
            
            return {
                "score": score,
                "distance_meters": distance,
                "nearest_eco_location": nearest,
                "user_coordinates": actual_location,
                "location_source": "image" if img_location else "user",
                "timestamp_valid": self._validate_timestamp(timestamp)
            }
            
        except Exception as e:
            return {
                "score": 0.0,
                "error": str(e)
            }
    
    def _validate_timestamp(self, timestamp: Optional[float]) -> bool:
        """Validate if timestamp is recent (within 24 hours)."""
        if timestamp is None:
            return True
            
        from time import time
        return (time() - timestamp) <= 86400  # 24 hours in seconds