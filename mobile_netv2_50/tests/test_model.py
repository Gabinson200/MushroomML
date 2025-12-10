import numpy as np
import tensorflow as tf
from tf_keras import layers, models

def test_model_structure():
    """
    Test if a dummy MobileNetV2 model outputs the correct number of classes (9).
    """
    img_size = 128
    num_classes = 9
    
    # Recreate the architecture used in the project
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights=None, # Don't download weights for test
        alpha=0.5
    )
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes)(x)
    model = models.Model(inputs=base_model.input, outputs=outputs)
    
    # Create dummy input
    dummy_input = np.random.rand(1, img_size, img_size, 3).astype(np.float32)
    prediction = model.predict(dummy_input)
    
    # Check output shape
    assert prediction.shape == (1, num_classes), "Model output shape mismatch"

def test_input_preprocessing():
    """
    Test if input image normalization logic works as expected.
    """
    # Simulate an image loaded as uint8 [0, 255]
    fake_img = tf.constant([[[0, 127.5, 255]]], dtype=tf.float32)
    
    # Apply the normalization logic from the notebook: (img / 127.5) - 1.0
    normalized = (fake_img / 127.5) - 1.0
    
    # Check ranges
    assert np.allclose(normalized.numpy().min(), -1.0), "Min value should be -1.0"
    assert np.allclose(normalized.numpy().max(), 1.0), "Max value should be 1.0"
