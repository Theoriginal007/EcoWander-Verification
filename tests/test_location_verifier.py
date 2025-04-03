import pytest
from ecowander.verification.location_verifier import LocationVerifier
from ecowander.config.eco_locations import KNOWN_ECO_LOCATIONS

@pytest.fixture
def location_verifier():
    return LocationVerifier()

class TestLocationVerifier:
    def test_verify_location(self, location_verifier):
        # Test with known location
        test_coords = KNOWN_ECO_LOCATIONS[0]['coordinates']
        result = location_verifier.verify_location(
            image_path=None,
            user_location=test_coords
        )
        assert result['score'] == 1.0
        assert result['distance_meters'] <= 100
    
    def test_far_location(self, location_verifier):
        # Test with location far from any known spot
        result = location_verifier.verify_location(
            image_path=None,
            user_location=(0.0, 0.0)  # Middle of nowhere
        )
        assert result['score'] < 0.5
    
    def test_invalid_input(self, location_verifier):
        with pytest.raises(ValueError):
            location_verifier.verify_location(
                image_path=None,
                user_location=None
            )