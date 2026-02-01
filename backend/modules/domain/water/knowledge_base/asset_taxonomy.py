"""Water infrastructure asset taxonomy and classifications."""

# Asset type hierarchy
ASSET_TAXONOMY = {
    "pipe": {
        "aliases": ["pipeline", "main", "line", "conduit"],
        "subtypes": ["transmission", "distribution", "service"],
        "materials": ["pvc", "ductile iron", "cast iron", "copper", "hdpe", "steel"],
    },
    "pump": {
        "aliases": ["pumping station", "booster"],
        "subtypes": ["centrifugal", "submersible", "vertical turbine", "booster"],
        "components": ["motor", "impeller", "shaft", "seal"],
    },
    "valve": {
        "aliases": ["gate", "control valve"],
        "subtypes": ["gate", "butterfly", "ball", "check", "air release", "prv", "psv"],
        "operations": ["open", "close", "throttle", "regulate"],
    },
    "meter": {
        "aliases": ["flow meter", "water meter"],
        "subtypes": ["magnetic", "ultrasonic", "turbine", "positive displacement"],
        "measurements": ["flow", "pressure", "consumption"],
    },
    "sensor": {
        "aliases": ["monitoring device", "transducer"],
        "subtypes": ["pressure", "flow", "level", "quality", "temperature"],
        "protocols": ["4-20ma", "modbus", "hart", "profibus"],
    },
    "tank": {
        "aliases": ["reservoir", "storage tank", "clearwell"],
        "subtypes": ["elevated", "ground", "standpipe"],
        "capacity_units": ["gallons", "cubic meters", "million gallons"],
    },
    "hydrant": {
        "aliases": ["fire hydrant", "flushing hydrant"],
        "subtypes": ["wet barrel", "dry barrel"],
        "operations": ["flush", "flow test", "paint"],
    },
}

# Location/zone types
ZONE_TYPES = ["pressure zone", "dma", "district", "sector", "area", "region"]

# Priority levels
PRIORITY_LEVELS = {
    "critical": ["emergency", "urgent", "immediate", "critical"],
    "high": ["high", "important", "priority"],
    "medium": ["medium", "normal", "routine"],
    "low": ["low", "deferred", "future"],
}


def normalize_asset_type(query_text: str) -> str:
    """Normalize asset type mentions in query to canonical form."""
    query_lower = query_text.lower()

    for asset_type, details in ASSET_TAXONOMY.items():
        # Check primary type
        if asset_type in query_lower:
            return asset_type

        # Check aliases
        for alias in details.get("aliases", []):
            if alias in query_lower:
                return asset_type

    return None


def get_asset_subtypes(asset_type: str) -> list:
    """Get valid subtypes for an asset."""
    return ASSET_TAXONOMY.get(asset_type, {}).get("subtypes", [])
