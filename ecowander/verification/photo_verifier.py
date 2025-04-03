import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow import lite as tflite
from typing import Dict, Optional, List
import logging
from datetime import datetime
from pathlib import Path
from ecowander.config.settings import MODEL_SETTINGS

class PhotoVerifier:
    """Verifies eco-actions in photos using TensorFlow Lite model."""
    
    def __init__(self, dummy_mode: bool = False):
        """
        Initialize the photo verifier.
        
        Args:
            dummy_mode: If True, uses mock verification for testing
        """
        self.logger = self._setup_logging()
        self.dummy_mode = dummy_mode
        
        if not dummy_mode:
            self.model, self.labels = self._initialize_model()
            self.input_details = self.model.get_input_details()
            self.output_details = self.model.get_output_details()
            self._log_initialization()

    def _setup_logging(self) -> logging.Logger:
        """Configure and return logger instance."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(handler)
        return logger

    def _initialize_model(self) -> tuple[tflite.Interpreter, List[str]]:
        """Load and validate model with labels."""
        try:
            model = self._load_model()
            labels = self._load_label_map()
            return model, labels
        except Exception as e:
            self.logger.error("Initialization failed: %s", str(e))
            raise RuntimeError(f"PhotoVerifier initialization failed: {str(e)}")

    def _load_model(self) -> tflite.Interpreter:
        """Load and validate TFLite model."""
        model_path = Path(MODEL_SETTINGS["model_path"])
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file missing at {model_path}")

        try:
            self.logger.info("Loading model from: %s", model_path)
            interpreter = tflite.Interpreter(model_path=str(model_path))
            
            # Validate model structure
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            
            if not input_details:
                raise ValueError("Model has no input tensors")
                
            expected_shape = (1, MODEL_SETTINGS["input_height"], 
                            MODEL_SETTINGS["input_width"], 3)
            if tuple(input_details[0]['shape']) != expected_shape:
                raise ValueError(f"Model expects input shape {expected_shape}")
                
            return interpreter
            
        except Exception as e:
            self.logger.error("Model loading failed: %s", str(e))
            raise

    def _load_label_map(self) -> List[str]:
        """Load and validate label map file."""
        label_path = Path(MODEL_SETTINGS["model_path"]).with_name("label_map.txt")
        
        if not label_path.exists():
            raise FileNotFoundError(f"Label file missing at {label_path}")

        try:
            with open(label_path, 'r') as f:
                labels = [line.strip().split(": ")[1] for line in f if ":" in line]
                
            if len(labels) != 5:
                raise ValueError(f"Expected 5 labels, got {len(labels)}")
                
            return labels
            
        except Exception as e:
            self.logger.error("Label loading failed: %s", str(e))
            raise

    def _log_initialization(self):
        """Log successful initialization details."""
        self.logger.info("PhotoVerifier initialized with %d classes", len(self.labels))
        print("\n[DEBUG] Model Initialization:")
        print(f"- Input Shape: {self.input_details[0]['shape']}")
        print(f"- Output Shape: {self.output_details[0]['shape']}")
        print(f"- Labels: {self.labels}")

    def verify_photo(self, image_path: str, challenge_type: Optional[str] = None) -> Dict:
        """
        Verify if photo shows valid eco-action.
        
        Args:
            image_path: Path to image file
            challenge_type: Specific eco-challenge being verified
            
        Returns:
            Dictionary with verification results
        """
        if self.dummy_mode:
            return self._dummy_verification(challenge_type)
            
        try:
            img_array = self._preprocess_image(image_path)
            predictions = self._run_inference(img_array)
            result = self._process_predictions(predictions)
            
            if challenge_type:
                result = self._apply_challenge_rules(result, challenge_type.lower(), image_path)
            
            return result
            
        except UnidentifiedImageError:
            error_msg = f"Invalid image file: {image_path}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Photo verification failed: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image for model input."""
        with Image.open(image_path) as img:
            if img.format not in ('JPEG', 'PNG'):
                raise ValueError(f"Unsupported image format: {img.format}")
                
            print(f"\n[DEBUG] Processing image: {image_path}")
            print(f"- Original: {img.size} pixels, {img.format}")
            
            img = img.convert('RGB')
            img_array = np.array(img.resize(
                (MODEL_SETTINGS["input_width"], MODEL_SETTINGS["input_height"])
            ), dtype=np.float32) / 255.0
            
            print(f"- Processed range: {np.min(img_array):.2f}-{np.max(img_array):.2f}")
            return np.expand_dims(img_array, axis=0)

    def _run_inference(self, img_array: np.ndarray) -> np.ndarray:
        """Run model inference on prepared image."""
        try:
            self.model.set_tensor(self.input_details[0]['index'], img_array)
            self.model.invoke()
            predictions = self.model.get_tensor(self.output_details[0]['index'])[0]
            
            print(f"[DEBUG] Raw predictions: {predictions}")
            if np.all(predictions == 0):
                raise ValueError("Model returned all zeros - possibly uninitialized")
                
            return predictions
            
        except Exception as e:
            raise RuntimeError(f"Inference failed: {str(e)}")

    def _process_predictions(self, predictions: np.ndarray) -> Dict:
        """Convert model predictions to verification results."""
        return {
            "predicted_class": self.labels[np.argmax(predictions)],
            "confidence": float(np.max(predictions)),
            "class_scores": {
                self.labels[i]: float(predictions[i]) 
                for i in range(len(self.labels))
            },
            "timestamp": datetime.now().isoformat(),
            "is_valid": False  # Default, updated by challenge rules
        }

    def _apply_challenge_rules(self, result: Dict, challenge_type: str, image_path: str) -> Dict:
        """Apply special rules for specific challenge types."""
        # Cherry blossom verification
        if "cherry_blossom" in challenge_type:
            result.update(self._verify_cherry_blossom(image_path))
        
        # Recycling verification
        elif "recycling" in challenge_type:
            result["is_valid"] = (
                result["predicted_class"] == "valid_recycling" and 
                result["confidence"] > 0.7
            )
        
        return result

    def _verify_cherry_blossom(self, image_path: str) -> Dict:
        """Special verification for cherry blossom challenge."""
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                pixels = np.array(img)
                
                # Detect pink pixels (adjusted for JPEG)
                is_pink = (
                    (pixels[:,:,0] > 180) &  # Red
                    (pixels[:,:,1] > 80) &   # Green
                    (pixels[:,:,2] > 120) &  # Blue
                    (pixels[:,:,0] > pixels[:,:,1] * 1.3)  # More red than green
                )
                pink_ratio = np.mean(is_pink)
                
                # Check if current date is in season (March 20 - April 15)
                today = datetime.now().date()
                seasonal = (today.month == 3 and today.day >= 20) or \
                          (today.month == 4 and today.day <= 15)
                
                return {
                    "pink_pixel_ratio": float(pink_ratio),
                    "seasonal_valid": seasonal,
                    "is_valid": seasonal and pink_ratio > 0.08
                }
                
        except Exception as e:
            self.logger.warning("Cherry blossom analysis failed: %s", str(e))
            return {}

    def _dummy_verification(self, challenge_type: Optional[str] = None) -> Dict:
        """Generate mock verification results for testing."""
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
            "class_scores": {cls: np.random.uniform(0, 0.3) for cls in dummy_classes},
            "timestamp": datetime.now().isoformat(),
            "is_valid": class_idx != 0
        }