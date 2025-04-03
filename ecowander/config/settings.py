import os
from pathlib import Path

# Model configuration
MODEL_SETTINGS = {
    "model_name": "eco_action_verifier",
    "model_path": str(Path(__file__).parent.parent.parent / "models" / "eco_action_model.tflite"),
    "input_width": 224,
    "input_height": 224
}

# Verification thresholds
VERIFICATION_THRESHOLDS = {
    "photo_min_confidence": 0.7,
    "location_max_distance": 100,  # meters
    "fraud_max_score": 0.5
}

# Application settings
APP_SETTINGS = {
    "debug": True,
    "max_image_size": 5 * 1024 * 1024,  # 5MB
    "allowed_extensions": [".jpg", ".jpeg", ".png"]
}

# Cherry blossom season (March 20 - April 15)
CHERRY_BLOSSOM_SEASON = {
    "start_month": 3,
    "start_day": 20,
    "end_month": 4,
    "end_day": 15
}