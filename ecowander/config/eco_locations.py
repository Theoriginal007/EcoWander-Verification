from ecowander.verification.models import EcoLocation
from typing import List

# Known eco-locations database
KNOWN_ECO_LOCATIONS: List[EcoLocation] = [
    EcoLocation(
        name="Tokyo Central Park Recycling Center",
        coordinates=(35.682839, 139.759455),
        radius_meters=50,
        challenge_types=["recycling", "waste_management"],
        description="Central recycling point with proper waste separation"
    ),
    EcoLocation(
        name="Kyoto Cherry Blossom Conservation Area",
        coordinates=(35.0116, 135.7681),
        radius_meters=200,
        challenge_types=["cherry_blossom", "nature_conservation"],
        description="Protected area for cherry blossom trees"
    ),
    EcoLocation(
        name="Osaka Eco Station",
        coordinates=(34.6937, 135.5023),
        radius_meters=30,
        challenge_types=["recycling", "eco_education"],
        description="Environmental education and recycling center"
    )
]

def get_locations_by_challenge(challenge_type: str) -> List[EcoLocation]:
    """Filter locations by supported challenge types."""
    return [
        loc for loc in KNOWN_ECO_LOCATIONS
        if challenge_type in loc.challenge_types
    ]