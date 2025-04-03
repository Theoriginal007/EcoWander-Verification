import pytest
from ecowander.verification.photo_verifier import PhotoVerifier
from ecowander.services.image_processor import process_image_for_model
import os

@pytest.fixture
def photo_verifier():
    return PhotoVerifier()

@pytest.fixture
def test_image_path():
    return os.path.join(os.path.dirname(__file__), "test_data", "test_image.jpg")

class TestPhotoVerifier:
    def test_initialization(self, photo_verifier):
        assert photo_verifier.model is not None
        assert photo_verifier.input_details is not None
        assert photo_verifier.output_details is not None
    
    def test_verify_photo(self, photo_verifier, test_image_path):
        confidence, details = photo_verifier.verify_photo(
            test_image_path,
            "test_challenge"
        )
        assert 0 <= confidence <= 1
        assert "model_used" in details
        assert "input_shape" in details
    
    def test_invalid_image(self, photo_verifier):
        with pytest.raises(ValueError):
            photo_verifier.verify_photo(
                "nonexistent.jpg",
                "test_challenge"
            )