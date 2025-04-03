from pydantic import BaseModel
from typing import Optional, Dict, Tuple, List

class VerificationRequest(BaseModel):
    image_path: str
    user_location: Tuple[float, float]
    challenge_type: str
    user_id: Optional[str] = None
    timestamp: Optional[float] = None
    metadata: Optional[Dict] = None

class VerificationResult(BaseModel):
    is_verified: bool
    overall_score: float
    photo_verification: Dict
    location_verification: Dict
    fraud_detection: Dict
    timestamp: str
    challenge_type: str
    
    class Config:
        json_encoders = {
            "float": lambda v: round(v, 4)
        }

class EcoLocation(BaseModel):
    name: str
    coordinates: Tuple[float, float]
    radius_meters: float
    challenge_types: List[str]
    description: Optional[str] = None

class EcoActionVerifier:
    def __init__(self):
        from .photo_verifier import PhotoVerifier
        from .location_verifier import LocationVerifier
        from .fraud_detector import FraudDetector
        
        self.photo_verifier = PhotoVerifier()
        self.location_verifier = LocationVerifier()
        self.fraud_detector = FraudDetector()

    def verify_eco_action(self, image_path, user_location, challenge_type):
        """Main verification method that combines all checks"""
        # Implementation would go here
        pass