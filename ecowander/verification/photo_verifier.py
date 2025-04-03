import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow import lite as tflite
from typing import Dict, Optional, List
import logging
from datetime import datetime
from pathlib import Path
from ecowander.config.settings import MODEL_SETTINGS

class PhotoVerifier:
    def __init__(self, dummy_mode: bool = False):
        self.logger = logging.getLogger(__name__)
        self.dummy_mode = dummy_mode
        
        if not dummy_mode:
            self.model = self._load_model()
            self.labels = self._load_label_map()
            self.input_details = self.model.get_input_details()
            self.output_details = self.model.get_output_details()
            self.logger.info("PhotoVerifier initialized with %d classes", len(self.labels))

    def _load_model(self) -> tflite.Interpreter:
        """Load and validate TFLite model"""
        model_path = Path(MODEL_SETTINGS["model_path"])
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file missing at {model_path}")

        try:
            self.logger.info("Loading model from: %s", model_path)
            interpreter = tflite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            
            # Validate model structure
            input_details = interpreter.get_input_details()
            if not input_details:
                raise ValueError("Model has no input tensors")
                
            self.logger.info("Model loaded (Input shape: %s)", input_details[0]['shape'])
            return interpreter
            
        except Exception as e:
            self.logger.error("Model loading failed: %s", str(e))
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def _load_label_map(self) -> List[str]:
        """Load and validate label map"""
        label_path = Path(MODEL_SETTINGS["model_path"]).with_name("label_map.txt")
        
        if not label_path.exists():
            raise FileNotFoundError(f"Label file missing at {label_path}")

        try:
            with open(label_path, 'r') as f:
                labels = [line.strip().split(": ")[1] for line in f if ":" in line]
                
            if len(labels) != 5:
                raise ValueError(f"Expected 5 labels, got {len(labels)}")
                
            self.logger.info("Loaded labels: %s", labels)
            return labels
            
        except Exception as e:
            self.logger.error("Label loading failed: %s", str(e))
            raise RuntimeError(f"Label loading failed: {str(e)}")

    def verify_photo(self, image_path: str, challenge_type: Optional[str] = None) -> Dict:
        """Verify photo against eco-action classes"""
        if self.dummy_mode:
            return self._dummy_verification(challenge_type)
            
        try:
            # Load and preprocess image
            with Image.open(image_path) as img:
                if img.format not in ('JPEG', 'PNG'):
                    raise ValueError("Only JPEG/PNG images supported")
                    
                img = img.convert('RGB')
                img_array = np.array(img.resize(
                    (MODEL_SETTINGS["input_width"], MODEL_SETTINGS["input_height"])
                ), dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

            # Run inference
            self.model.set_tensor(self.input_details[0]['index'], img_array)
            self.model.invoke()
            predictions = self.model.get_tensor(self.output_details[0]['index'])[0]
            
            # Process results
            result = {
                "predicted_class": self.labels[np.argmax(predictions)],
                "confidence": float(np.max(predictions)),
                "class_scores": {
                    self.labels[i]: float(predictions[i]) 
                    for i in range(len(self.labels))
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Apply challenge-specific rules
            if challenge_type:
                result = self._apply_challenge_rules(result, challenge_type.lower(), image_path)
            
            return result
            
        except UnidentifiedImageError:
            self.logger.error("Invalid image file: %s", image_path)
            raise ValueError("Invalid or corrupted image file")
        except Exception as e:
            self.logger.error("Verification failed: %s", str(e))
            raise RuntimeError(f"Photo verification failed: {str(e)}")

    def _apply_challenge_rules(self, result: Dict, challenge_type: str, image_path: str) -> Dict:
        """Apply special rules for cherry blossom verification"""
        if "cherry_blossom" in challenge_type:
            try:
                with Image.open(image_path) as img:
                    img = img.convert('RGB')
                    pixels = np.array(img)
                    
                    # Pink pixel detection
                    is_pink = (
                        (pixels[:,:,0] > 180) &  # Red
                        (pixels[:,:,1] > 80) &   # Green
                        (pixels[:,:,2] > 120) &  # Blue
                        (pixels[:,:,0] > pixels[:,:,1] * 1.3)  # More red than green
                    )
                    pink_ratio = np.mean(is_pink)
                    
                    # Seasonal check (March 20 - April 15)
                    today = datetime.now().date()
                    seasonal = (today.month == 3 and today.day >= 20) or \
                              (today.month == 4 and today.day <= 15)
                    
                    # Adjust confidence
                    if seasonal:
                        result["confidence"] = min(1.0, result["confidence"] + 0.15)
                    if pink_ratio > 0.08:
                        result["confidence"] = min(1.0, result["confidence"] + (pink_ratio * 0.5))
                    
                    result.update({
                        "seasonal_valid": seasonal,
                        "pink_pixel_ratio": float(pink_ratio),
                        "is_valid": result["predicted_class"] == "cherry_blossom_activity"
                    })
                    
            except Exception as e:
                self.logger.warning("Cherry blossom analysis failed: %s", str(e))
        
        return result

    def _dummy_verification(self, challenge_type: Optional[str] = None) -> Dict:
        """Generate mock verification results"""
        dummy_classes = [
            "invalid_action",
            "valid_recycling", 
            "valid_composting",
            "valid_conservation",
            "cherry_blossom_activity"
        ]
        
        class_idx = 4 if challenge_type and "cherry" in challenge_type.lower() else np.random.randint(0, 5)
        confidence = np.random.uniform(0.6, 0.95)
        
        return {
            "predicted_class": dummy_classes[class_idx],
            "confidence": confidence,
            "is_valid": class_idx != 0,
            "class_scores": {cls: np.random.uniform(0, 0.3) for cls in dummy_classes},
            "timestamp": datetime.now().isoformat()
        }