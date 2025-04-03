import hashlib
from PIL import Image
from ecowander.services.hashing_service import (
    generate_image_hash,
    check_image_manipulation
)
from typing import Dict, Optional

class FraudDetector:
    def __init__(self):
        self.known_hashes = set()
        
    def detect_fraud(
        self,
        image_path: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Detect potential fraud in submitted images.
        
        Args:
            image_path: Path to the image file
            user_id: Optional user identifier
            metadata: Additional submission metadata
            
        Returns:
            Dictionary with fraud detection results
        """
        try:
            # Generate image hash
            img_hash = generate_image_hash(image_path)
            
            # Check for duplicates
            is_duplicate = img_hash in self.known_hashes
            if not is_duplicate:
                self.known_hashes.add(img_hash)
            
            # Check for manipulation
            manipulation_result = check_image_manipulation(image_path)
            
            # Calculate fraud score (0 = clean, 1 = high fraud risk)
            fraud_score = 0.0
            if is_duplicate:
                fraud_score = 0.9
            elif manipulation_result["is_edited"]:
                fraud_score = max(0.5, fraud_score + 0.4)
            
            return {
                "fraud_score": fraud_score,
                "image_hash": img_hash,
                "is_duplicate": is_duplicate,
                "manipulation_detected": manipulation_result,
                "user_id": user_id,
                "metadata": metadata
            }
            
        except Exception as e:
            return {
                "fraud_score": 0.5,  # Default to medium risk if error
                "error": str(e)
            }