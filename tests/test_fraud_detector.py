import pytest
from ecowander.verification.fraud_detector import FraudDetector
import os

@pytest.fixture
def fraud_detector():
    return FraudDetector()

@pytest.fixture
def test_image_path():
    return os.path.join(os.path.dirname(__file__), "test_data", "test_image.jpg")

class TestFraudDetector:
    def test_detect_fraud(self, fraud_detector, test_image_path):
        result = fraud_detector.detect_fraud(test_image_path)
        assert 0 <= result['fraud_score'] <= 1
        assert isinstance(result['image_hash'], str)
        assert len(result['image_hash']) > 0
    
    def test_duplicate_detection(self, fraud_detector, test_image_path):
        # First submission
        result1 = fraud_detector.detect_fraud(test_image_path)
        assert result1['is_duplicate'] is False
        
        # Second submission with same image
        result2 = fraud_detector.detect_fraud(test_image_path)
        assert result2['is_duplicate'] is True
        assert result2['fraud_score'] > 0.8