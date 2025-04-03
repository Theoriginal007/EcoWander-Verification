# scripts/validate_model.py
import tensorflow as tf
import os
import logging
from typing import Optional

def validate_tflite_model(model_path: str) -> bool:
    """
    Validates a TensorFlow Lite model file.
    Returns True if valid, False otherwise.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Basic file checks
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return False

        file_size = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"Model found at {model_path} ({file_size:.2f} MB)")

        # Try loading with available interpreters
        try:
            # First try with LiteRT if available
            from ai_edge_litert import Interpreter as LiteRTInterpreter
            logger.info("Testing with LiteRT interpreter...")
            interpreter = LiteRTInterpreter(model_path=model_path)
            interpreter.allocate_tensors()
        except ImportError:
            # Fall back to legacy interpreter
            logger.info("LiteRT not available, testing with TF Lite interpreter...")
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()

        # Check model specifications
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logger.info("\n=== Model Specifications ===")
        logger.info(f"Input shape: {input_details[0]['shape']}")
        logger.info(f"Input type: {input_details[0]['dtype']}")
        logger.info(f"Output shape: {output_details[0]['shape']}")
        logger.info(f"Output type: {output_details[0]['dtype']}")

        return True

    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="../models/eco_action_model.tflite",
                      help="Path to the tflite model file")
    args = parser.parse_args()

    is_valid = validate_tflite_model(args.model_path)
    print(f"\nValidation result: {'SUCCESS' if is_valid else 'FAILED'}")
    exit(0 if is_valid else 1)