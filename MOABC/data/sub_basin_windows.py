"""
Sub-basin rectangular windows for GPM IMERG areal precipitation extraction.
Windows defined by reservoir locations and CDR2 gauge control sections.
"""

SUB_BASINS = [
    {
        "name": "Wudongde",
        "lat_min": 25.5, "lat_max": 27.0,
        "lon_min": 100.5, "lon_max": 102.8,
        "dam_lat": 26.3, "dam_lon": 102.6,
        "cdr2_gauge": 60101300,
    },
    {
        "name": "Baihetan",
        "lat_min": 26.5, "lat_max": 28.0,
        "lon_min": 102.0, "lon_max": 103.2,
        "dam_lat": 27.0, "dam_lon": 102.9,
        "cdr2_gauge": 60102100,
    },
    {
        "name": "Xiluodu",
        "lat_min": 27.5, "lat_max": 29.0,
        "lon_min": 102.5, "lon_max": 104.0,
        "dam_lat": 28.3, "dam_lon": 103.7,
        "cdr2_gauge": 60104100,
    },
    {
        "name": "Xiangjiaba",
        "lat_min": 28.0, "lat_max": 29.5,
        "lon_min": 103.5, "lon_max": 105.0,
        "dam_lat": 28.7, "dam_lon": 104.4,
        "cdr2_gauge": 60104800,
    },
    {
        "name": "TGD",
        "lat_min": 29.0, "lat_max": 31.5,
        "lon_min": 104.0, "lon_max": 112.0,
        "dam_lat": 30.8, "dam_lon": 111.0,
        "cdr2_gauge": 60107170,
    },
]
