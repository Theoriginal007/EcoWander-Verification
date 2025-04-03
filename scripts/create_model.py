import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from pathlib import Path

def create_and_save_model():
    # Create output directory
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # 1. Create a simple CNN model (lightweight for TFLite)
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(224, 224, 3)),
        keras.layers.Rescaling(1./255),  # Normalize pixel values
        
        # Feature extraction
        keras.layers.Conv2D(16, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        
        # Classifier
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(5, activation='softmax')  # 5 output classes
    ])
    
    # 2. Compile with dummy optimizer (not needed for conversion)
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    # 3. Convert to TensorFlow Lite with proper settings
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optimize for size
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Standard TFLite ops
        tf.lite.OpsSet.SELECT_TF_OPS      # Fallback to TF ops if needed
    ]
    
    # 4. Convert and save
    tflite_model = converter.convert()
    model_path = model_dir / "eco_action_model.tflite"
    with open(model_path, 'wb') as f:
        f.write(tflite_model)
    
    # 5. Create label map
    labels = [
        "0: invalid_action",
        "1: valid_recycling",
        "2: valid_composting", 
        "3: valid_conservation",
        "4: cherry_blossom_activity"
    ]
    with open(model_dir / "label_map.txt", 'w') as f:
        f.write("\n".join(labels))
    
    print(f"Successfully created model at: {model_path}")
    print("Model Specifications:")
    print("- Input: 224x224 RGB image (0-255 values)")
    print("- Output: 5-class probability distribution")
    print("- Classes:", [label.split(": ")[1] for label in labels])

if __name__ == "__main__":
    create_and_save_model()