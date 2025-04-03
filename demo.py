#!/usr/bin/env python3
"""
EcoWander Verification System - Demonstration Script
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from ecowander.verification import EcoActionVerifier

def main():
    print("EcoWander Action Verification System")
    print("===================================\n")
    
    try:
        # 1. Verify required files and folders
        required_dirs = ["models", "demo_images"]
        for dir_name in required_dirs:
            if not Path(dir_name).exists():
                print(f"Error: Missing directory '{dir_name}'")
                return

        # 2. Check model files
        model_path = Path("models/eco_action_model.tflite")
        label_path = Path("models/label_map.txt")
        
        if not model_path.exists():
            print(f"Error: Model file not found at {model_path}")
            return
            
        if not label_path.exists():
            print(f"Error: Label file not found at {label_path}")
            return

        # 3. Check test image - UPDATED TO MATCH YOUR FILENAME
        demo_image = Path("demo_images/cherry_blossom.jpeg")
        if not demo_image.exists():
            print(f"Error: Test image not found at {demo_image}")
            print("Available images in demo_images/:")
            for img in Path("demo_images").iterdir():
                print(f"- {img.name}")
            return

        print("✓ All required files found")
        print(f"Using image: {demo_image}")
        
        # 4. Initialize verifier
        print("\nInitializing verifier...")
        verifier = EcoActionVerifier()
        print("✓ Verifier initialized successfully")

        # 5. Run verification
        print("\nStarting verification...")
        results = verifier.verify_eco_action(
            image_path=str(demo_image),
            user_location=(35.682839, 139.759455),
            challenge_type="cherry_blossom"
        )
        
        if results is None:
            print("Error: Verification returned None")
            return
            
        # 6. Display results
        print("\nVERIFICATION SUCCESSFUL")
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        print(f"\n!! SYSTEM ERROR: {str(e)}", file=sys.stderr)
        if 'debug' in globals() and debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()