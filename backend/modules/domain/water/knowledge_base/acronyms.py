"""Water infrastructure acronyms and abbreviations dictionary."""

# Common water infrastructure acronyms
WATER_ACRONYMS = {
    "SCADA": "Supervisory Control and Data Acquisition",
    "DMA": "District Metered Area",
    "PRV": "Pressure Reducing Valve",
    "PSV": "Pressure Sustaining Valve",
    "WTP": "Water Treatment Plant",
    "WWTP": "Wastewater Treatment Plant",
    "AMR": "Automated Meter Reading",
    "AMI": "Advanced Metering Infrastructure",
    "GIS": "Geographic Information System",
    "CMMS": "Computerized Maintenance Management System",
    "NRW": "Non-Revenue Water",
    "MNF": "Minimum Night Flow",
    "ILI": "Infrastructure Leakage Index",
    "PLC": "Programmable Logic Controller",
    "RTU": "Remote Terminal Unit",
    "HMI": "Human Machine Interface",
    "VFD": "Variable Frequency Drive",
    "GPM": "Gallons Per Minute",
    "PSI": "Pounds per Square Inch",
    "MGD": "Million Gallons per Day",
    "CIP": "Capital Improvement Program",
    "O&M": "Operations and Maintenance",
    "AWWA": "American Water Works Association",
}

# Maintenance code prefixes
MAINTENANCE_CODE_PREFIXES = {
    "PM": "Preventive Maintenance",
    "CM": "Corrective Maintenance",
    "EM": "Emergency Maintenance",
    "IN": "Inspection",
    "RE": "Repair",
    "RP": "Replacement",
    "CL": "Cleaning",
    "TE": "Testing",
    "CA": "Calibration",
}


def expand_acronym(text: str) -> str:
    """Expand acronyms in text for better query understanding."""
    expanded = text
    for acronym, full_form in WATER_ACRONYMS.items():
        expanded = expanded.replace(acronym, f"{acronym} ({full_form})")
    return expanded
