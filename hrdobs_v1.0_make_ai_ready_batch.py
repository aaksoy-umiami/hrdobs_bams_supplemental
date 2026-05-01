# =============================================================================
# hrdobs_v1.0_make_ai_ready_batch.py
# HRDOBS v1.0 — AI-Ready Dataset Conversion Script
#
# Converts original HRDOBS HDF5 files into a standardized "AI-ready" format
# suitable for use in data assimilation, machine learning, and research
# applications.  Each output file has a flat group structure, standardized
# units, observation error estimates, CF-compliant time encoding, and a
# companion Kerchunk JSON sidecar for virtual (lazy) access.
#
# USAGE
# -----
# Run interactively and select one of seven operating modes at the prompt:
#
#   Mode 1 — Identify Double Entries
#             Scan input files for NHC codes that map to multiple storm names
#             within the same year and report the conflicts.
#
#   Mode 2 — Rename Double-Entry Files
#             Preview a rename plan that resolves double-entry conflicts to a
#             single canonical name, then optionally execute it.
#
#   Mode 3 — Check Temporal Gaps
#             For every storm in the input directory, build a unified
#             deduplicated 6-hourly cycle timeline and report any gaps.
#
#   Mode 4 — Spline Track Altitude Diagnostic
#             Determine the flight-level pressure altitude for each spline
#             track from raw .fix files.  Produces a diagnostic CSV used by
#             Mode 6 to inject the correct pressure into the output files.
#
#   Mode 5 — QC Scan (Dry Run)
#             Run the full quality-control pipeline on all input files without
#             writing any output.  Reports how many files would be skipped and
#             the expected error-assignment actions.
#
#   Mode 6 — Full Processing
#             Main conversion pipeline.  For each input file: applies QC,
#             flattens the group hierarchy, standardizes units, assigns
#             observation errors, converts time to CF convention, writes the
#             AI-ready HDF5 file and its Kerchunk JSON sidecar, then produces
#             a dataset inventory CSV and schema reference CSV.
#
#   Mode 7 — Rebuild DB and Schema Only
#             Re-reads existing AI-ready files to regenerate the inventory CSV
#             and schema reference CSV without re-converting any source data.
#
# KEY CONFIGURATION (top of script)
# ----------------------------------
#   INPUT_DIR          — directory containing original HRDOBS HDF5 files
#   OUTPUT_DIR         — root directory for AI-ready output files
#   INFO_DIR           — auxiliary information directory
#   BASIN_FILTER       — 'ATL', 'EPAC', or 'ALL'
#   SHIPS_CSV_PATH     — path to SHIPS predictor CSV (from ships_to_csv.py)
#   BANNED_INSTRUMENTS — instrument types excluded from output
#
# SUBROUTINE SUMMARY
# ------------------
# Configuration and metadata helpers
#   process_root_metadata        Clean and normalize raw HDF5 global attributes
#   load_ships_lookup            Load SHIPS predictor CSV into a keyed lookup dict
#   _ships_key_from_hrdobs       Derive the SHIPS lookup key from an HRDOBS filename
#   detect_basin                 Identify storm basin (ATL / EPAC) from storm_id
#   should_process_file          Filter files by the configured BASIN_FILTER
#   extract_filename_metadata    Parse storm metadata encoded in the filename
#
# Fix-file and spline track helpers
#   find_fix_file                Locate the raw .fix file for a given storm
#   parse_fix_file               Parse a .fix file into a list of fix records
#   compute_spline_altitude      Compute mean flight-level altitude from fix records
#   extract_flight_level_pressure  Extract flight-level pressure from source HDF5
#
# Data utilities
#   decode_attr                  Decode any HDF5 attribute value to a plain Python type
#   safe_attr                    Convert values to h5py-compatible strings or scalars
#   extract_error_val            Extract a leading numeric value from an error attribute
#   replace_missing_values       Replace sentinel/fill values with NaN
#   convert_packed_time_to_cf    Convert YYYYMMDDHHMMSS floats to CF seconds since epoch
#   generate_virtual_manifest    Generate a Kerchunk JSON sidecar for an HDF5 file
#
# Quality control
#   resolve_tc_category          Resolve raw NHC tc_category to a standardized token
#   _apply_time_qc               Apply calendar-parse and file-window time QC in-place
#   _check_time_span             Advisory check for suspiciously short time spans
#   validate_and_clean_data      Full QC pipeline: bounds, anchor, all-NaN, location
#
# Core conversion
#   convert_universal            Convert one source HDF5 file to AI-ready format
#   extract_inventory_and_schema Re-read an AI-ready file to rebuild inventory/schema
#   save_schema                  Write the dataset schema reference CSV
#
# Pre-conversion diagnostics
#   identify_double_entries      Report NHC codes with multiple storm names (Mode 1)
#   canonical_name               Select canonical name from competing storm IDs
#   check_temporal_gaps          Report missing 6-hourly cycles per storm (Mode 3)
#   rename_double_entries        Preview / execute storm-name normalization (Mode 2)
#   check_spline_track_altitudes Produce the spline altitude diagnostic (Mode 4)
#
# Entry point
#   main                         Interactive mode selector and top-level dispatcher
#
# AUTHORS
#   Kathryn Sellwood, Altug Aksoy, Brittany Dahl
#   NOAA / AOML / HRD
#
# DATASET VERSION
#   HRDOBS v1.0
# =============================================================================

import os
import glob
import h5py
import numpy as np
import pandas as pd
import json
import re
import shutil
import datetime
import csv
import warnings
import kerchunk.hdf
import ujson
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
# All directory paths and output filenames are defined here.  Change these
# to relocate inputs, outputs, or reports without touching any other code.

# --- Input / output directories ---
INPUT_DIR  = "HRDOBS_hdf5"              # Original HRDOBS HDF5 source files
INFO_DIR   = "HRDOBS_info"              # Auxiliary information directory
OUTPUT_DIR = "AI_ready_dataset"         # Root directory for AI-ready output
FIXES_DIR  = "fixes_for_spline_tracks"  # Raw .fix files for spline track altitudes

# --- Input files ---
SHIPS_CSV_PATH = "ships_converted_for_hrdobs.csv"  # SHIPS predictor CSV (from ships_to_csv.py)

# --- Output report filenames ---
SPLINE_ALT_REPORT      = "spline_track_altitude_report.csv"
SPLINE_GAP_DIAGNOSTICS = "spline_track_gap_diagnostics.txt"
QC_FORENSICS_REPORT    = "qc_forensics_report.csv"
ERROR_SIM_REPORT       = "error_assignment_simulation.csv"
SHIPS_MISMATCH_REPORT  = "ships_metadata_mismatches.csv"
SHIPS_NOMATCH_REPORT   = "ships_no_match_cycles.csv"
INVENTORY_DB           = "hrdobs_inventory_db.csv"
SCHEMA_REPORT          = "hrdobs_dataset_schema.csv"
DOUBLE_ENTRIES_REPORT  = "hrdobs_double_entries.txt"
TEMPORAL_GAPS_REPORT   = "hrdobs_temporal_gaps.txt"
RENAME_PLAN_REPORT     = "hrdobs_rename_plan.txt"

BANNED_INSTRUMENTS = ['lidar', 'dwl', 'coyote']

# SHIPS parameters to explicitly extract into the inventory CSV database
INVENTORY_SHIPS_VARS = ['incv_kt', 'dtl_km', 'shrd_kt', 'shtd_deg', 'rhmd_pct', 'nsst_degc', 'nohc_kjcm2', 'vmpi_kt']

# =============================================================================
# REQUIRED METADATA FIELDS
# =============================================================================
# All 23 global attributes expected in every AI-ready file.  Divided into:
#
# CRITICAL — must be present with a real value (not 'NaN', not empty).
#   If any critical field is NaN or missing, the file is skipped entirely.
#
# EXPECTED — must be present and non-empty.  If missing, logged as a QC
#   issue but the file is still written.

CRITICAL_METADATA = {
    'storm_name', 'storm_id', 'storm_datetime', 'storm_epoch',
    'storm_intensity_ms', 'storm_mslp_hpa', 'tc_category',
    'center_from_tc_vitals', 'radius_of_maximum_wind_km',
}

# Exhaustive list of possible observation groups based on the dataset schema.
EXPECTED_GROUPS_LIST = [
    'dropsonde_ghawk', 'dropsonde_noaa42', 'dropsonde_noaa43', 'dropsonde_noaa49', 'dropsonde_usaf',
    'flight_level_hdobs_noaa42', 'flight_level_hdobs_noaa43', 'flight_level_hdobs_noaa49', 'flight_level_hdobs_usaf',
    'sfmr_noaa42', 'sfmr_noaa43', 'sfmr_usaf', 'ships_params',
    'tdr_noaa42', 'tdr_noaa43', 'tdr_noaa49',
    'track_best_track', 'track_spline_track', 'track_vortex_message'
]

EXPECTED_METADATA = {
    'creator_email', 'creator_name',
    'geospatial_lat_max', 'geospatial_lat_min',
    'geospatial_lat_units',
    'geospatial_lon_max', 'geospatial_lon_min',
    'geospatial_lon_units',
    'existing_groups', 'expected_groups', 'storm_motion_speed_kt', 'storm_motion_heading_deg',
    'time_coverage_start', 'time_coverage_end',
    'title', 'version_number',
}

# Hardcoded sentinel values to treat as missing (applied as fallback if the
# group's 'missing_value' attribute is absent or unparseable).
SENTINEL_VALUES = {-99.0, -999.0, 9999.0, 999.0, 99.0}

# Variables that are "core anchors" — if any of these are NaN after missing-
# value replacement the entire row is deleted.
CORE_ANCHOR_VARS = {'lat', 'latitude', 'lon', 'longitude', 'time'}

# Variables that are coordinate/geometric fields — never assigned an error field.
# Mirrors COORD_VARS in hrdobs_v1.0_validate_ai_ready_batch.py; keep both in sync.
COORD_VARS = {
    'time', 'lat', 'latitude', 'lon', 'longitude',
    'ght',          # geopotential height  (vertical coordinate)
    'p',            # pressure             (vertical coordinate)
    'az',           # TDR azimuth angle    (beam geometry)
    'elev',         # TDR elevation angle  (beam geometry)
    'clat', 'clon', # track center position
    'rr',           # SFMR rain rate (used to derive spd error, not an assimilation ob)
}

# Track product variables — not direct assimilation obs, no error field required.
TRACK_PRODUCT_VARS = {'vmax', 'pmin', 'rmw', 'sfcp', 'sfcspd', 'sfcdir'}

# Error fallback table: used only when no error array exists in the file AND
# no 'error_estimate_for_data_assimilation' attribute is present on the dataset.
# Keyed on variable name only — applies regardless of platform or instrument type.
ERROR_FALLBACK_RULES = {
    'rvel': 2.0,   # TDR radial velocity         (m/s)
    'u':    2.0,   # zonal wind                   (m/s)
    'v':    2.0,   # meridional wind              (m/s)
    'w':    0.5,   # vertical wind                (m/s)
    't':    0.5,   # temperature                  (K)
    'p':  100.0,   # pressure                     (Pa)
}

# =============================================================================
# OBSERVATION ERROR CONFIGURATION
# =============================================================================
OBS_ERROR_CONFIG = {
    # 1. Standard Flight Level
    'flight_level_hdobs_noaa42': {
        'terr': {'type': 'CONSTANT', 'value': 0.5},
        'uerr': {'type': 'CONSTANT', 'value': 2.0},
        'verr': {'type': 'CONSTANT', 'value': 2.0},
        'perr': {'type': 'CONSTANT', 'value': 100.0},
        'qerr': {'type': 'DYNAMIC_FALLBACK', 'value': 0.0005},
    },
    'flight_level_hdobs_noaa43': {
        'terr': {'type': 'CONSTANT', 'value': 0.5},
        'uerr': {'type': 'CONSTANT', 'value': 2.0},
        'verr': {'type': 'CONSTANT', 'value': 2.0},
        'perr': {'type': 'CONSTANT', 'value': 100.0},
        'qerr': {'type': 'DYNAMIC_FALLBACK', 'value': 0.0005},
    },
    'flight_level_hdobs_usaf': {
        'terr': {'type': 'CONSTANT', 'value': 0.5},
        'uerr': {'type': 'CONSTANT', 'value': 2.0},
        'verr': {'type': 'CONSTANT', 'value': 2.0},
        'perr': {'type': 'CONSTANT', 'value': 100.0},
        'qerr': {'type': 'DYNAMIC_FALLBACK', 'value': 0.0005},
    },
    
    # 2. High-Altitude Flight Level
    'flight_level_hdobs_noaa49': {
        'terr': {'type': 'CONSTANT', 'value': 1.25},
        'uerr': {'type': 'CONSTANT', 'value': 5.0},
        'verr': {'type': 'CONSTANT', 'value': 5.0},
        'perr': {'type': 'CONSTANT', 'value': 250.0},
        'qerr': {'type': 'DYNAMIC_FALLBACK', 'value': 0.00125},
    },
    
    # 3. Dropsondes
    'dropsonde_ghawk': {
        'terr': {'type': 'CONSTANT', 'value': 0.5},
        'uerr': {'type': 'CONSTANT', 'value': 2.0},
        'verr': {'type': 'CONSTANT', 'value': 2.0},
        'werr': {'type': 'CONSTANT', 'value': 0.5},
        'perr': {'type': 'CONSTANT', 'value': 100.0},
        'qerr': {'type': 'DYNAMIC_FALLBACK', 'value': 0.0005},
    },
    'dropsonde_noaa42': {
        'terr': {'type': 'CONSTANT', 'value': 0.5},
        'uerr': {'type': 'CONSTANT', 'value': 2.0},
        'verr': {'type': 'CONSTANT', 'value': 2.0},
        'werr': {'type': 'CONSTANT', 'value': 0.5},
        'perr': {'type': 'CONSTANT', 'value': 100.0},
        'qerr': {'type': 'DYNAMIC_FALLBACK', 'value': 0.0005},
    },
    'dropsonde_noaa43': {
        'terr': {'type': 'CONSTANT', 'value': 0.5},
        'uerr': {'type': 'CONSTANT', 'value': 2.0},
        'verr': {'type': 'CONSTANT', 'value': 2.0},
        'werr': {'type': 'CONSTANT', 'value': 0.5},
        'perr': {'type': 'CONSTANT', 'value': 100.0},
        'qerr': {'type': 'DYNAMIC_FALLBACK', 'value': 0.0005},
    },
    'dropsonde_usaf': {
        'terr': {'type': 'CONSTANT', 'value': 0.5},
        'uerr': {'type': 'CONSTANT', 'value': 2.0},
        'verr': {'type': 'CONSTANT', 'value': 2.0},
        'werr': {'type': 'CONSTANT', 'value': 0.5},
        'perr': {'type': 'CONSTANT', 'value': 100.0},
        'qerr': {'type': 'DYNAMIC_FALLBACK', 'value': 0.0005},
    },
    'dropsonde_noaa49': {
        'terr': {'type': 'CONSTANT', 'value': 0.5},
        'uerr': {'type': 'CONSTANT', 'value': 2.0},
        'verr': {'type': 'CONSTANT', 'value': 2.0},
        'werr': {'type': 'CONSTANT', 'value': 0.5},
        'perr': {'type': 'CONSTANT', 'value': 100.0},
        'qerr': {'type': 'DYNAMIC_FALLBACK', 'value': 0.00125},
    },
    
    # 4. TDR
    'tdr_noaa42': {
        'rvelerr': {'type': 'CONSTANT', 'value': 2.0},
    },
    'tdr_noaa43': {
        'rvelerr': {'type': 'CONSTANT', 'value': 2.0},
    },
    'tdr_noaa49': {
        'rvelerr': {'type': 'CONSTANT', 'value': 2.0},
    },
    
    # 5. SFMR
    'sfmr_noaa42': {
        'spderr': {'type': 'DYNAMIC_FALLBACK', 'value': 5.0},
    },
    'sfmr_noaa43': {
        'spderr': {'type': 'DYNAMIC_FALLBACK', 'value': 5.0},
    },
    'sfmr_usaf': {
        'spderr': {'type': 'DYNAMIC_FALLBACK', 'value': 5.0},
    },
}

# Physical bounds for QC (applied after missing-value masking).
# Units must match what is stored in the file.
VALID_BOUNDS = {
    'lat':       (-90.0,    90.0),
    'latitude':  (-90.0,    90.0),   
    'lon':       (-180.0,  180.0),
    'longitude': (-180.0,  180.0),   
    'time':      (19900000000000.0, 20300000000000.0),  # YYYYMMDDHHMMSS
    'sfcp':      (80000.0, 110000.0),                   # Pa
    'height':    (0.0, 20000.0),
    'altitude':  (0.0, 20000.0),
    'ght':       (0.0, 20000.0),
}

# NHC tc_category codes that are valid as-is (no further resolution needed).
# HU is intentionally excluded — it requires intensity to resolve to H1-H5.
NHC_PASSTHROUGH_CATS = {'TD', 'TS', 'SS', 'SD', 'EX', 'LO', 'DB', 'WV'}

# =============================================================================
# DATASET-LEVEL METADATA CONSTANTS
# =============================================================================
# These are written into every AI-ready file — never copied from source.
CREATOR_EMAIL   = 'kathryn.sellwood@noaa.gov, altug.aksoy@noaa.gov, brittany.dahl@noaa.gov'
CREATOR_NAME    = 'Kathryn Sellwood, Altug Aksoy, Brittany Dahl'
TITLE           = 'HRDOBS Consolidated AI-Ready TC Reconnaissance Observations'
VERSION_NUMBER  = 'v1.0'

# Geospatial bounding box half-width (degrees) around the storm center.
# Written into the file's global metadata attributes only — not used to
# filter observation data (see OBS_PROXIMITY_DEG below for that).
GEOSPATIAL_HALFWIDTH = 10.0

# Maximum allowed great-circle distance (expressed as a simple lat/lon
# box half-width in degrees) between an observation and the storm center at
# cycle time.  Observations outside this box are deleted and logged.
OBS_PROXIMITY_DEG = 10.0

# Metadata attributes that are ALWAYS computed by the conversion — never
# passed through from the source file, even if they exist there.
CONTROLLED_ATTRS = {
    'creator_email', 'creator_name', 'title', 'version_number',
    'existing_groups', 'expected_groups',
    'geospatial_lat_max', 'geospatial_lat_min', 'geospatial_lat_units',
    'geospatial_lon_max', 'geospatial_lon_min', 'geospatial_lon_units',
    'time_coverage_start', 'time_coverage_end',
    'storm_motion',
}

# =============================================================================
# BASIN FILTERING
# =============================================================================
# Which storm basins to process.  Options:
#   'ATL'  — Atlantic only  (storm IDs ending in …L, e.g. BERYL02L)
#   'EPAC' — East Pacific only (storm IDs ending in …E, e.g. NEWTON15E)
#   'ALL'  — both basins
BASIN_FILTER = 'ATL'

# Basin suffix mapping — maps the trailing letter(s) of the NHC code to a
# human-readable basin name.
_BASIN_SUFFIX_MAP = {
    'L': 'ATL',
    'E': 'EPAC',
}
_BASIN_LABEL = {'ATL': 'Atlantic', 'EPAC': 'East Pacific', 'ALL': 'All Basins'}

# =============================================================================
# SPLINE TRACK FIX FILES (Mode 7)
# =============================================================================
# Directory containing raw fix files used to derive spline tracks.
FIXES_DIR = "fixes_for_spline_tracks"

# Output filename for the spline track altitude report (Mode 4).
SPLINE_ALT_REPORT = "spline_track_altitude_report.csv"

# Aircraft to exclude from flight-level pressure extraction.
FL_EXCLUDED_AIRCRAFT = {'giv', 'g-iv', 'g_iv', 'noaa49', 'noaa_giv'}

# =============================================================================
# SHIPS PREDICTOR CSV
# =============================================================================
# Path to the SHIPS CSV file produced by ships_to_csv.py.
SHIPS_CSV_PATH = "ships_converted_for_hrdobs.csv"

# Tolerances for flagging discrepancies between HRDOBS metadata and the
# corresponding SHIPS HEAD values.  These are advisory only — no data is
# modified.  The comparison is performed in natural units.
SHIPS_MISMATCH_TOL_LAT  = 0.5   # degrees
SHIPS_MISMATCH_TOL_LON  = 0.5   # degrees
SHIPS_MISMATCH_TOL_VMAX = 2.5   # kt  (HRDOBS m/s converted before comparison)
SHIPS_MISMATCH_TOL_MSLP = 1.0   # hPa

# Metadata (units, long_name) for each SHIPS predictor column written to HDF5.
# Must stay in sync with TARGET_VARS in ships_to_csv.py.
SHIPS_PREDICTOR_META = {
    'type':           ('categorical', 'Storm type (0=wave/remnant/dissipating, 1=tropical, 2=subtropical, 3=extratropical)'),
    'incv_kt':        ('kt',          'Intensity change -6 to 0 hr'),
    'csst_degc':      ('degC',        'Climatological SST along track'),
    'cd20_m':         ('m',           'Climatological depth of 20 deg C isotherm'),
    'cd26_m':         ('m',           'Climatological depth of 26 deg C isotherm'),
    'cohc_kjcm2':     ('kJ/cm2',      'Climatological ocean heat content relative to 26 deg C isotherm'),
    'dtl_km':         ('km',          'Distance to nearest major land mass'),
    'oage_hr':        ('hr',          'Ocean age (time storm occupied area within 100 km)'),
    'nage_hr':        ('hr',          'Intensity-weighted ocean age'),
    'shrd_kt':        ('kt',          '850-200 hPa shear magnitude (r=200-800 km)'),
    'shtd_deg':       ('deg',         'Heading of 850-200 hPa shear vector (90=westerly)'),
    'shdc_kt':        ('kt',          '850-200 hPa vortex-removed shear magnitude (r=0-500 km)'),
    'sddc_deg':       ('deg',         'Heading of vortex-removed shear vector'),
    'rhlo_pct':       ('%',           '850-700 hPa relative humidity (r=200-800 km)'),
    'rhmd_pct':       ('%',           '700-500 hPa relative humidity (r=200-800 km)'),
    'rhhi_pct':       ('%',           '500-300 hPa relative humidity (r=200-800 km)'),
    'vmpi_kt':        ('kt',          'Maximum potential intensity (Kerry Emanuel)'),
    'penv_hpa':       ('hPa',         'Average environmental surface pressure (r=200-800 km)'),
    'penc_hpa':       ('hPa',         'Azimuthally averaged surface pressure at outer vortex edge'),
    'z850_1e7_per_s': ('1e-7 s-1',    '850 hPa vorticity (r=0-1000 km); divide by 1e7 for SI units'),
    'd200_1e7_per_s': ('1e-7 s-1',    '200 hPa divergence; divide by 1e7 for SI units'),
    'u200_kt':        ('kt',          '200 hPa zonal wind (r=200-800 km); negative=easterly'),
    'dsst_degc':      ('degC',        'Daily Reynolds SST along track'),
    'nsst_degc':      ('degC',        'NCODA analysis SST'),
    'nohc_kjcm2':     ('kJ/cm2',      'NCODA ocean heat content relative to 26 deg C isotherm'),
}

# =============================================================================
# HELPERS
# =============================================================================

def process_root_metadata(source_attrs):
    """
    Takes a dictionary of raw attributes and returns a cleaned dictionary 
    with native numerical types and split arrays for the AI-ready dataset.
    """
    cleaned = {}

    # Guarantee these fields always exist even if storm_motion is absent.
    cleaned['storm_motion_speed_kt'] = np.nan
    cleaned['storm_motion_heading_deg'] = np.nan
    
    # Define keys that require a direct cast to float
    float_keys = [
        'radius_of_maximum_wind_km', 'storm_intensity_ms', 'storm_mslp_hpa',
        'geospatial_lat_max', 'geospatial_lat_min', 
        'geospatial_lon_max', 'geospatial_lon_min'
    ]

    for k, v in source_attrs.items():
        # Handle cases where attributes might remain as bytes
        if isinstance(v, bytes):
            v = v.decode('utf-8', errors='ignore')

        # 1. Cast direct floats
        if k in float_keys:
            try:
                cleaned[k] = float(v)
            except (ValueError, TypeError):
                cleaned[k] = np.nan
                
        # 2. Cast epoch to integer
        elif k == 'storm_epoch':
            try:
                cleaned[k] = int(v)
            except (ValueError, TypeError):
                cleaned[k] = v
                
        # 3. Split center coordinates into a native float array
        elif k == 'center_from_tc_vitals':
            if isinstance(v, str) and ',' in v:
                lat_str, lon_str = v.split(',')
                cleaned[k] = [float(lat_str.strip()), float(lon_str.strip())]
            else:
                cleaned[k] = v
                
        # 4. Split comma-separated lists into native string arrays
        elif k in ('existing_groups', 'expected_groups', 'platforms'):
            if isinstance(v, str):
                cleaned[k] = [p.strip() for p in v.split(',') if p.strip()]
            elif isinstance(v, list):
                cleaned[k] = v
            else:
                cleaned[k] = v
                
        # 5. Split storm_motion into speed and heading floats
        elif k == 'storm_motion':
            if isinstance(v, str):
                match = re.search(r'(\d+(?:\.\d+)?)\s*kts?.*?(\d+(?:\.\d+)?)\s*deg', v, re.IGNORECASE)
                if match:
                    cleaned['storm_motion_speed_kt'] = float(match.group(1))
                    cleaned['storm_motion_heading_deg'] = float(match.group(2))
                else:
                    cleaned[k] = v # Retain original string if unparseable
            else:
                cleaned[k] = v
        
        # 6. Pass everything else through unchanged
        else:
            cleaned[k] = v
            
    return cleaned


def load_ships_lookup(csv_path):
    """
    Load a SHIPS predictor CSV produced by ships_to_csv.py and return a
    lookup dict keyed by (basin_suffix, storm_num, year_str, datetime_utc).

    The key deliberately avoids storm name because SHIPS truncates names to
    4 characters (e.g. 'ARTH') while HRDOBS uses full names (e.g. 'ARTHUR').
    Matching on basin suffix + storm number + year + cycle datetime is
    unambiguous and robust across all naming variants.

    Returns an empty dict if the file does not exist or cannot be read.
    """
    if not csv_path or not os.path.isfile(csv_path):
        print(f"  ⚠️  SHIPS CSV not found: {csv_path} "
              f"— ships_params group will not be written.")
        return {}

    _BASIN_TO_SUFFIX = {
        'AL': 'L', 'EP': 'E', 'CP': 'C',
        'WP': 'W', 'IO': 'I', 'SH': 'S',
    }
    lookup  = {}
    skipped = 0

    try:
        with open(csv_path, newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(
                line for line in fh if not line.startswith('#')
            )
            for row in reader:
                atcf_id      = row.get('atcf_id', '').strip()
                datetime_utc = row.get('datetime_utc', '').strip()
                if len(atcf_id) < 8 or not datetime_utc:
                    skipped += 1
                    continue
                basin_suffix = _BASIN_TO_SUFFIX.get(atcf_id[:2].upper(), '?')
                storm_num    = atcf_id[2:4]
                year_str     = atcf_id[4:8]
                key          = (basin_suffix, storm_num, year_str, datetime_utc)
                lookup[key]  = row
    except Exception as e:
        print(f"  ⚠️  Could not load SHIPS CSV ({e}) "
              f"— ships_params group will not be written.")
        return {}

    suffix = f" ({skipped} rows skipped)" if skipped else ""
    print(f"  📌 SHIPS predictors loaded: {len(lookup):,} cycles{suffix}")
    return lookup


def _ships_key_from_hrdobs(storm_id, storm_datetime):
    """
    Derive the SHIPS lookup key (basin_suffix, storm_num, year_str,
    datetime_utc) from an HRDOBS storm_id (e.g. 'ARTH01L') and
    storm_datetime (e.g. '2014-07-01T00:00:00Z').

    Storm_id pattern: <name_alpha><2-digit_number><basin_suffix_letter(s)>
    e.g. 'ARTH01L', 'GONZALO08L', 'TWO02L'

    Returns None if the storm_id cannot be parsed.
    """
    m = re.match(r'^[A-Za-z]+?(\d+)([A-Za-z]+)$', storm_id)
    if not m:
        return None
    storm_num    = m.group(1).zfill(2)
    basin_suffix = m.group(2)[-1].upper()   # last letter is basin suffix
    year_str     = storm_datetime[:4] if len(storm_datetime) >= 4 else ''
    if not year_str:
        return None
    return (basin_suffix, storm_num, year_str, storm_datetime)


def detect_basin(storm_id):
    """
    Detect the basin from a storm_id string.

    Extracts the trailing letter(s) of the NHC code (e.g. 'L' from 'BERYL02L',
    'E' from 'NEWTON15E') and maps it to a basin label using _BASIN_SUFFIX_MAP.

    Returns 'ATL', 'EPAC', or 'UNKNOWN'.
    """
    m = re.search(r'\d+([A-Z]+)$', storm_id.upper())
    if m:
        return _BASIN_SUFFIX_MAP.get(m.group(1), 'UNKNOWN')
    return 'UNKNOWN'


def should_process_file(filepath):
    """
    Check whether a file should be processed under the current BASIN_FILTER.

    Extracts storm_id from the filename, detects its basin, and returns True
    only if the basin matches the configured BASIN_FILTER (or filter is 'ALL').
    """
    meta = extract_filename_metadata(filepath)
    if not meta:
        return False   # unparseable filename — skip
    storm_id = meta.get('storm_id', '')
    if not storm_id:
        return False
    basin = detect_basin(storm_id)
    if BASIN_FILTER == 'ALL':
        return True
    return basin == BASIN_FILTER


def find_fix_file(storm_id, year):
    """
    Locate the raw .fix file for a given storm.

    Expected path:
        FIXES_DIR / Fixes_<YYYY> / <YYYY>_<n>.fix
    where <n> is the alphabetic prefix of storm_id in title case.
    e.g. storm_id='BERYL02L', year=2024 → fixes_for_spline_tracks/Fixes_2024/2024_Beryl.fix

    Returns the full path if the file exists, otherwise None.
    """
    name_match = re.match(r'([A-Za-z]+)', storm_id)
    if not name_match:
        return None
    name = name_match.group(1).title()   # BERYL → Beryl
    fix_filename = f"{year}_{name}.fix"
    fix_path = os.path.join(FIXES_DIR, f"Fixes_{year}", fix_filename)
    if os.path.isfile(fix_path):
        return fix_path
    return None


def parse_fix_file(filepath):
    """
    Parse a .fix file and return a list of non-excluded fix records.

    Each record is a dict with keys:
        dt       — datetime object (UTC)
        lat      — float (degrees N positive)
        lon      — float (degrees E positive / W negative)
        alt_mb   — float (pressure altitude in mb)
        source   — str   (aircraft callsign, e.g. 'NOAA43', 'USAF')

    The file contains three sections:
        1. Fixes (center positions) — parsed
        2. Passes (eye-wall crossings) — skipped
        3. Excluded fixes — skipped (already excluded from spline fit)

    Time rollover correction: some fix-file generators produce timestamps
    like 11:60:00 (seconds=60) or 12:60:00 (minutes=60) due to an
    off-by-one bug.  These are corrected automatically:
        seconds == 60 → +1 minute, seconds = 0
        minutes == 60 → +1 hour,   minutes = 0
    """
    fixes = []

    in_fixes = False
    in_passes = False
    in_excluded = False

    def _correct_time(date_str, time_str):
        """Parse a date + time string, auto-correcting rollover fields."""
        hh, mm, ss = [int(x) for x in time_str.split(':')]
        carry = 0
        if ss >= 60:
            carry, ss = divmod(ss, 60)
            mm += carry
        if mm >= 60:
            carry, mm = divmod(mm, 60)
            hh += carry
        # hh overflow (>=24) is theoretically possible but extremely
        # unlikely; handle it via timedelta just in case.
        day_carry = 0
        if hh >= 24:
            day_carry, hh = divmod(hh, 24)
        dt = datetime.datetime.strptime(date_str, "%m/%d/%Y")
        dt = dt.replace(hour=hh, minute=mm, second=ss)
        if day_carry:
            dt += datetime.timedelta(days=day_carry)
        return dt

    with open(filepath, 'r') as fh:
        for line in fh:
            line = line.rstrip()
            # Detect section boundaries
            if 'fixes for' in line.lower() and 'excluded' not in line.lower():
                in_fixes = True;  in_passes = False; in_excluded = False
                continue
            if 'passes for' in line.lower():
                in_fixes = False; in_passes = True;  in_excluded = False
                continue
            if 'excluded' in line.lower() and 'fixes for' in line.lower():
                in_fixes = False; in_passes = False; in_excluded = True
                continue
            if line.strip().startswith('Date') or line.strip().startswith('MM/DD'):
                continue
            if not line.strip():
                continue

            # Only parse the non-excluded fixes section
            if not in_fixes:
                continue

            m = re.match(
                r'\s*(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2}:\d{2})\s+'
                r'([\d.]+)\s+([NS])\s+([\d.]+)\s+([EW])\s+'
                r'([\d.]+)\s+(\S+)',
                line
            )
            if m:
                dt  = _correct_time(m.group(1), m.group(2))
                lat = float(m.group(3)) * (1 if m.group(4) == 'N' else -1)
                lon = float(m.group(5)) * (-1 if m.group(6) == 'W' else 1)
                alt = float(m.group(7))
                src = m.group(8)
                # Skip the BestTrack entry (alt=0) that sometimes appears as
                # the last fix
                if alt > 0:
                    fixes.append({
                        'dt': dt, 'lat': lat, 'lon': lon,
                        'alt_mb': alt, 'source': src,
                    })

    return fixes


def compute_spline_altitude(fixes, spline_start_dt, spline_end_dt):
    """
    Given a list of parsed fixes and a spline track time window,
    return the mean altitude (mb) of all fixes within the window.

    Returns:
        mean_alt    — float or None if no fixes in window
        n_fixes     — int, number of fixes in window
        alt_counts  — dict {altitude_mb: count}
    """
    from collections import Counter

    in_window = [f for f in fixes
                 if spline_start_dt <= f['dt'] <= spline_end_dt]

    if not in_window:
        return None, 0, {}

    alts = [f['alt_mb'] for f in in_window]
    alt_counts = dict(Counter(alts))
    mean_alt = sum(alts) / len(alts)

    return mean_alt, len(in_window), alt_counts


def extract_flight_level_pressure(f_in):
    """
    Scan an ORIGINAL HDF5 file for flight-level observation groups and
    extract the pressure ('p') variable from each.

    Original file structure (3-level):
        <top_key> / <mid_key> / <platform_key> / p
    e.g.:
        flight level / hdobs / NOAA43 / p

    Returns a dict:
        {
            'groups':       list of str  — flat group names found
            'per_aircraft': dict         — {aircraft: {'n_obs': int,
                                                       'mean_p_mb': float,
                                                       'min_p_mb': float,
                                                       'max_p_mb': float}}
            'overall_mean_p_mb':  float or None
            'overall_n_obs':      int
        }

    Pressure values > 2000 are assumed to be in Pa and converted to mb
    (÷100).  Values that are NaN after missing-value replacement are
    excluded.
    """
    result = {
        'groups': [],
        'per_aircraft': {},
        'overall_mean_p_mb': None,
        'overall_n_obs': 0,
    }

    all_pressures = []

    for top_key in f_in.keys():
        top_key_l = top_key.lower()
        if top_key_l in ('track', 'storm stats'):
            continue

        for mid_key in f_in[top_key].keys():
            mid_grp = f_in[top_key][mid_key]
            if not isinstance(mid_grp, h5py.Group):
                continue

            # Detect depth: peek at first child
            first_child = next(iter(mid_grp.values()), None)
            if first_child is None:
                continue

            if isinstance(first_child, h5py.Dataset):
                # 2-level: mid_grp IS the leaf
                # Check if this looks like a flight-level group with 'p'
                mid_key_l = mid_key.lower().replace(" ", "")
                if 'p' in mid_grp:
                    flat_name = mid_key.replace(" ", "_")
                    if 'flight' not in mid_key_l:
                        continue
                    # Skip excluded aircraft (e.g. G-IV / NOAA 49)
                    if any(ex in mid_key_l for ex in FL_EXCLUDED_AIRCRAFT):
                        continue
                    p_raw = mid_grp['p'][:]
                    if np.issubdtype(p_raw.dtype, np.number):
                        grp_missing = mid_grp.attrs.get('missing_value', None)
                        p_arr = replace_missing_values(p_raw, grp_missing)
                        p_valid = p_arr[~np.isnan(p_arr)]
                        if len(p_valid) > 0:
                            # Convert Pa → mb if needed
                            if np.nanmean(p_valid) > 2000:
                                p_valid = p_valid / 100.0
                            result['groups'].append(flat_name)
                            result['per_aircraft'][flat_name] = {
                                'n_obs':      len(p_valid),
                                'mean_p_mb':  round(float(np.nanmean(p_valid)), 1),
                                'min_p_mb':   round(float(np.nanmin(p_valid)), 1),
                                'max_p_mb':   round(float(np.nanmax(p_valid)), 1),
                            }
                            all_pressures.extend(p_valid.tolist())
            else:
                # 3-level: iterate platforms inside mid_grp
                mid_key_l = mid_key.lower()
                if 'flight' not in mid_key_l:
                    continue
                for plat_key in mid_grp.keys():
                    plat_key_l = plat_key.lower().replace(" ", "")
                    # Skip excluded aircraft (e.g. G-IV / NOAA 49)
                    if any(ex in plat_key_l for ex in FL_EXCLUDED_AIRCRAFT):
                        continue
                    plat_grp = mid_grp[plat_key]
                    if not isinstance(plat_grp, h5py.Group):
                        continue
                    if 'p' not in plat_grp:
                        continue

                    flat_name = f"{mid_key}_{plat_key}".replace(" ", "_")
                    p_raw = plat_grp['p'][:]
                    if np.issubdtype(p_raw.dtype, np.number):
                        grp_missing = (plat_grp.attrs.get('missing_value', None)
                                       or mid_grp.attrs.get('missing_value', None))
                        p_arr = replace_missing_values(p_raw, grp_missing)
                        p_valid = p_arr[~np.isnan(p_arr)]
                        if len(p_valid) > 0:
                            # Convert Pa → mb if needed
                            if np.nanmean(p_valid) > 2000:
                                p_valid = p_valid / 100.0
                            result['groups'].append(flat_name)
                            result['per_aircraft'][flat_name] = {
                                'n_obs':      len(p_valid),
                                'mean_p_mb':  round(float(np.nanmean(p_valid)), 1),
                                'min_p_mb':   round(float(np.nanmin(p_valid)), 1),
                                'max_p_mb':   round(float(np.nanmax(p_valid)), 1),
                            }
                            all_pressures.extend(p_valid.tolist())

    if all_pressures:
        result['overall_mean_p_mb'] = round(float(np.mean(all_pressures)), 1)
        result['overall_n_obs'] = len(all_pressures)

    return result


def resolve_tc_category(raw_cat, intensity_ms):
    """
    Resolve a raw NHC tc_category string to a standardized category token.

    Resolution rules (applied in order):
      1. Passthrough categories (TD, TS, SS, SD, EX, LO, DB, WV) → returned as-is.
      2. 'HU' with a valid intensity → resolved to H1–H5 via Saffir-Simpson.
      3. 'HU' without intensity → resolved via intensity alone if available,
         otherwise returned as 'HU' (unknown tier).
      4. Missing/unparseable category but valid intensity → derived from
         intensity using Saffir-Simpson (TD/TS/H1-H5).
      5. Both missing → returns 'NaN'.

    Parameters
    ----------
    raw_cat      : str or None  — raw tc_category value from source file
    intensity_ms : float or str — wind speed in m/s ('NaN' string or float)

    Returns
    -------
    resolved : str  — standardized category token
    source   : str  — 'source' | 'derived' | 'unknown' (for QC logging)
    """
    def _intensity_to_cat(v_ms):
        """Convert m/s wind speed to Saffir-Simpson or tropical category."""
        v_kt = float(v_ms) * 1.94384
        if v_kt >= 137: return 'H5'
        if v_kt >= 113: return 'H4'
        if v_kt >= 96:  return 'H3'
        if v_kt >= 83:  return 'H2'
        if v_kt >= 64:  return 'H1'
        if v_kt >= 34:  return 'TS'
        return 'TD'

    # Normalize intensity
    try:
        v_ms = float(intensity_ms)
        has_intensity = not np.isnan(v_ms)
    except (TypeError, ValueError):
        has_intensity = False

    # Normalize category
    cat = str(raw_cat).strip().upper() if raw_cat not in (None, 'NaN', '', 'nan') else None

    if cat in NHC_PASSTHROUGH_CATS:
        return cat, 'source'

    if cat == 'HU':
        if has_intensity:
            return _intensity_to_cat(v_ms), 'derived'
        return 'HU', 'source'   # tier unknown — no intensity available

    # Category absent or unrecognized — try to derive from intensity
    if has_intensity:
        return _intensity_to_cat(v_ms), 'derived'

    return 'NaN', 'unknown'

def decode_attr(val):
    """Decode a scalar or 1-element array attribute to a plain Python value.

    Multi-element string arrays (e.g. storm_motion stored as
    [b' 19kts', b'290deg']) are joined into a single comma-separated string
    so no elements are silently dropped.
    """
    if isinstance(val, (np.ndarray, list)):
        if len(val) == 0:
            return ''
        # Multi-element string array — join all elements
        if len(val) > 1:
            parts = []
            for elem in val:
                if isinstance(elem, (bytes, np.bytes_)):
                    parts.append(elem.decode('utf-8', errors='ignore').rstrip('\x00').strip())
                else:
                    parts.append(str(elem).strip())
            return ', '.join(p for p in parts if p)
        val = val[0]
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode('utf-8', errors='ignore').rstrip('\x00').strip()
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def safe_attr(val):
    """
    Convert any HDF5 attribute value to a form that h5py writes as a
    variable-length UTF-8 string (for strings) or a plain Python numeric
    scalar (for numbers).

    This eliminates fixed-length H5T_STR_NULLTERM strings in the output,
    which are the root cause of the STRPAD truncation problem reported by
    readers that stop at the first null byte or whitespace in padded strings.

    Multi-element string arrays are joined into a single comma-separated
    string so no elements are silently dropped (same logic as decode_attr).

    Numeric values are preserved as int/float so they are stored as HDF5
    numeric scalars, not as string attributes.
    """
    if isinstance(val, (np.ndarray, list)):
        if len(val) == 0:
            return ''
        # Multi-element array — join if strings, take first if numeric
        if len(val) > 1:
            # Peek at first element to decide
            first = val[0]
            if isinstance(first, (bytes, np.bytes_, str)):
                parts = []
                for elem in val:
                    if isinstance(elem, (bytes, np.bytes_)):
                        parts.append(elem.decode('utf-8', errors='ignore').rstrip('\x00').strip())
                    else:
                        parts.append(str(elem).strip())
                return ', '.join(p for p in parts if p)
            # Numeric array — not expected as a scalar attr, take first element
            val = first
        else:
            val = val[0]
    # Numeric numpy scalars -> plain Python numerics (stored as HDF5 numbers)
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    # Bytes (fixed-length HDF5 strings) -> decoded plain Python str,
    # stripping null-byte padding that NULLTERM strings carry.
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode('utf-8', errors='ignore').rstrip('\x00').strip()
    # Already a plain Python str/int/float -> return as-is
    return val


def extract_error_val(attr_val):
    """Extract the leading numeric value from an error attribute like b'2.0 m/s'."""
    try:
        s = decode_attr(attr_val)
        match = re.search(r'([0-9]*\.?[0-9]+)', str(s))
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return None


def extract_filename_metadata(filepath):
    filename = os.path.basename(filepath)
    meta = {}
    
    match = re.search(r'HRDOBS_(.*?)[\._](\d{4})(\d{2})(\d{2})(\d{2})(\d{2})?', filename)
    if match:
        storm_id   = match.group(1)
        nm         = re.match(r'([A-Za-z]+)', storm_id)
        storm_name = nm.group(1).upper() if nm else storm_id
        
        year, month, day, hour = match.group(2), match.group(3), match.group(4), match.group(5)
        minute = match.group(6) if match.group(6) else "00"
        
        iso_str = f"{year}-{month}-{day}T{hour}:{minute}:00Z"
        dt = datetime.datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%SZ")
        dt = dt.replace(tzinfo=datetime.timezone.utc)
        
        meta['storm_id']       = storm_id
        meta['storm_name']     = storm_name
        meta['storm_datetime'] = iso_str
        meta['storm_epoch']    = int(dt.timestamp())
    return meta


def convert_packed_time_to_cf(packed_time_array):
    """
    Converts YYYYMMDDhhmmss.0 floats to seconds since 1900-01-01 00:00:00Z.
    Handles NaN values gracefully.
    """
    cf_time_array = np.full_like(packed_time_array, np.nan, dtype=np.float64)
    
    valid_mask = ~np.isnan(packed_time_array)
    if not valid_mask.any():
        return cf_time_array

    valid_times = packed_time_array[valid_mask]
    
    time_strs = valid_times.astype(np.int64).astype(str)
    time_strs = np.char.zfill(time_strs, 14)

    dt_objects = pd.to_datetime(time_strs, format='%Y%m%d%H%M%S', errors='coerce', utc=True)
    
    epoch = pd.Timestamp('1900-01-01 00:00:00Z', tz='UTC')
    
    seconds_since = (dt_objects - epoch).total_seconds()
    
    cf_time_array[valid_mask] = seconds_since
    return cf_time_array


def generate_virtual_manifest(hdf5_filepath):
    """
    Generates a Zarr-compatible virtual manifest (Kerchunk) for an HDF5 file.
    Creates a lightweight .json sidecar file in the same directory.
    """
    manifest_path = hdf5_filepath.replace('.hdf5', '.json')
    
    try:
        # Extract the B-tree byte ranges without copying the actual array data
        with open(hdf5_filepath, 'rb') as f:
            h5chunks = kerchunk.hdf.SingleHdf5ToZarr(f, hdf5_filepath)
            manifest = h5chunks.translate()
            
        # Write the manifest using ujson for speed
        with open(manifest_path, 'w') as outf:
            outf.write(ujson.dumps(manifest))
            
        return manifest_path
    except Exception as e:
        print(f"  ⚠️  Could not generate virtual manifest for {os.path.basename(hdf5_filepath)}: {e}")
        return None


# =============================================================================
# MISSING VALUE REPLACEMENT
# =============================================================================

def replace_missing_values(arr, group_missing_val=None):
    """
    Replace sentinel/missing values with NaN.

    Priority:
      1. The group-level 'missing_value' attribute (if parseable).
      2. The hardcoded SENTINEL_VALUES set as a safety net.

    Returns a float64 copy of the array.
    """
    arr = arr.astype(np.float64)

    # Build the set of values to mask
    sentinels_to_apply = set(SENTINEL_VALUES)  # start with hardcoded fallback

    if group_missing_val is not None:
        try:
            mv = float(str(decode_attr(group_missing_val)).strip())
            # Use the attribute value as the primary sentinel; keep hardcoded
            # list as additional safety net (they don't conflict in practice).
            sentinels_to_apply.add(mv)
        except (ValueError, TypeError):
            pass  # unparseable — fall back to hardcoded list only

    for sv in sentinels_to_apply:
        arr[arr == sv] = np.nan

    return arr


# =============================================================================
# QC
# =============================================================================

def _apply_time_qc(df_clean, ac, group_name, cycle_dt, window_hours, logs):
    """
    Apply time QC Layers 1 and 2 in-place to a time column.

    Layer 1 — Calendar parse: NaN any value that does not form a valid
    YYYYMMDDHHMMSS calendar date/time (e.g. month=60, hour=99).

    Layer 2 — File-window consistency: NaN any value that lies more than
    window_hours from the cycle time extracted from the filename.  This
    catches timestamps that are numerically plausible but belong to a
    completely different date/time than the file's observation window.

    Both layers are vectorised via pd.to_datetime for performance.
    Results are logged; anchor_mask is updated by the caller after return.

    Parameters
    ----------
    df_clean     : DataFrame being cleaned (modified in-place)
    ac           : actual column name in df_clean
    group_name   : for log entries
    cycle_dt     : datetime.datetime at cycle time (UTC); Layer 2 skipped if None
    window_hours : half-width of the acceptable time window around cycle_dt
    logs         : list to append QC log entries to
    """
    notna_mask = df_clean[ac].notna()
    if not notna_mask.any():
        return

    raw_vals = df_clean.loc[notna_mask, ac]

    # Vectorised parse — invalid calendar dates produce NaT
    parsed = pd.to_datetime(
        raw_vals.astype(int).astype(str).str.zfill(14),
        format='%Y%m%d%H%M%S',
        errors='coerce',
        utc=True
    )

    # --- Layer 1: calendar validity ---
    bad_l1 = parsed.isna()
    if bad_l1.any():
        bad_idx  = bad_l1[bad_l1].index
        bad_vals = df_clean.loc[bad_idx, ac].tolist()
        df_clean.loc[bad_idx, ac] = np.nan
        count = len(bad_idx)
        logs.append({
            'Group': group_name, 'Variable': ac,
            'Issue_Type': (
                f'Time QC Layer 1 — {count} value(s) failed calendar parse '
                f'(invalid YYYYMMDDHHMMSS): '
                f'{[int(v) for v in bad_vals[:10]]}'
                + (f' ...(+{count - 10} more)' if count > 10 else '')
            ),
            'Bad_Obs_Valid': count, 'Bad_Obs_Artifact': 0,
            'Total_Obs_Valid': len(df_clean), 'Total_Obs_Artifact': 0,
        })

    # --- Layer 2: file-window consistency ---
    if cycle_dt is not None:
        good_parsed = parsed[~bad_l1]
        if len(good_parsed) > 0:
            cycle_ts = pd.Timestamp(cycle_dt)
            diffs    = (good_parsed - cycle_ts).abs().dt.total_seconds()
            bad_l2   = diffs > window_hours * 3600
            if bad_l2.any():
                bad_idx  = bad_l2[bad_l2].index
                bad_vals = df_clean.loc[bad_idx, ac].tolist()
                df_clean.loc[bad_idx, ac] = np.nan
                count = len(bad_idx)
                logs.append({
                    'Group': group_name, 'Variable': ac,
                    'Issue_Type': (
                        f'Time QC Layer 2 — {count} value(s) outside '
                        f'±{window_hours}h of cycle time '
                        f'({cycle_dt.strftime("%Y-%m-%dT%H:%M:%SZ")}): '
                        f'{[int(v) for v in bad_vals[:10]]}'
                        + (f' ...(+{count - 10} more)' if count > 10 else '')
                    ),
                    'Bad_Obs_Valid': count, 'Bad_Obs_Artifact': 0,
                    'Total_Obs_Valid': len(df_clean), 'Total_Obs_Artifact': 0,
                })


def _check_time_span(df, group_name, filename, qc_logs, min_span_minutes=15):
    """
    Layer 3 — Minimum time span advisory.

    If the group has more than one observation but the span of valid,
    parseable timestamps is less than min_span_minutes, log an advisory.
    No data is modified.  Single-observation groups are exempt.
    """
    if len(df) < 2:
        return

    cl = {c.lower(): c for c in df.columns}
    time_col = cl.get('time')
    if not time_col:
        return

    valid_times = df[time_col].dropna()
    if len(valid_times) < 2:
        return

    parsed = pd.to_datetime(
        valid_times.astype(int).astype(str).str.zfill(14),
        format='%Y%m%d%H%M%S',
        errors='coerce'
    ).dropna()

    if len(parsed) < 2:
        return

    span_seconds = (parsed.max() - parsed.min()).total_seconds()
    if span_seconds < min_span_minutes * 60:
        mins = int(span_seconds // 60)
        secs = int(span_seconds % 60)
        qc_logs.append({
            'Filename':   filename,
            'Group':      group_name,
            'Variable':   'time',
            'Issue_Type': (
                f'Advisory (Time Layer 3): time span is only {mins}m {secs}s '
                f'across {len(parsed)} observations — timestamps may be '
                f'systematically corrupt (expected ≥ {min_span_minutes}m)'
            ),
            'Bad_Obs_Valid':    0,
            'Bad_Obs_Artifact': 0,
            'Total_Obs_Valid':  len(parsed),
            'Total_Obs_Artifact': 0,
        })


def validate_and_clean_data(df, group_name, cycle_dt=None):
    """
    Strict QC logic:
      1. Replace out-of-bounds values for known variables with NaN.
      2. Delete any row where a core anchor (lat/lon/time) is NaN.
         For time columns, also applies:
           Layer 1 — calendar parse check (rejects e.g. month=60)
           Layer 2 — file-window check (rejects times > ±4h from cycle)
      3. Delete any row where ALL observation variables are NaN.
      4. Rows that have *some* NaN obs but valid anchors are kept as-is.
      5. Delete group if any location column is entirely NaN.

    cycle_dt : datetime.datetime (UTC) extracted from the filename.
               If provided, Layer 2 time QC is applied.  Pass None to
               skip Layer 2 (e.g. scan-only mode without a parseable filename).

    Returns (df_clean, qc_entries).
    """
    logs     = []
    df_clean = df.copy()
    cols_lower = {c: c.lower() for c in df_clean.columns}

    # --- Step 1: Physical bounds QC (in-place NaN on individual bad values) ---
    bound_log = {}
    for col, col_l in cols_lower.items():
        if col_l in VALID_BOUNDS and col_l not in CORE_ANCHOR_VARS:
            lo, hi = VALID_BOUNDS[col_l]
            bad = (df_clean[col] < lo) | (df_clean[col] > hi)
            if bad.any():
                count = int(bad.sum())
                df_clean.loc[bad, col] = np.nan
                bound_log[col] = count

    for col, count in bound_log.items():
        logs.append({
            'Group': group_name, 'Variable': col,
            'Issue_Type': f'Bound Violation → NaN ({count} values)',
            'Bad_Obs_Valid': count, 'Bad_Obs_Artifact': 0,
            'Total_Obs_Valid': len(df_clean), 'Total_Obs_Artifact': 0
        })

    # --- Step 2: Delete rows with any NaN anchor ---
    anchor_cols = [c for c, cl in cols_lower.items() if cl in CORE_ANCHOR_VARS]
    anchor_mask = pd.Series(True, index=df_clean.index)
    for ac in anchor_cols:
        # Also enforce bounds on anchor columns.
        # VALID_BOUNDS includes both short-form ('lat', 'lon') and
        # long-form ('latitude', 'longitude') keys so this check fires
        # regardless of how the source file names its coordinate columns.
        ac_l = cols_lower[ac]
        if ac_l in VALID_BOUNDS:
            lo, hi = VALID_BOUNDS[ac_l]
            bad_anchor = (df_clean[ac] < lo) | (df_clean[ac] > hi)
            df_clean.loc[bad_anchor, ac] = np.nan
        # Time-specific QC: Layers 1 (calendar parse) and 2 (window check).
        # Applied after numeric bounds so already-NaN values are skipped.
        # Obs groups use ±4h; track groups call _apply_time_qc directly.
        if ac_l == 'time':
            _apply_time_qc(df_clean, ac, group_name, cycle_dt,
                           window_hours=4, logs=logs)
        anchor_mask &= df_clean[ac].notna()

    dropped_anchor = int((~anchor_mask).sum())
    if dropped_anchor > 0:
        logs.append({
            'Group': group_name, 'Variable': 'ANCHOR',
            'Issue_Type': f'Row Deleted — NaN/bad anchor ({dropped_anchor} rows)',
            'Bad_Obs_Valid': 0, 'Bad_Obs_Artifact': dropped_anchor,
            'Total_Obs_Valid': int(anchor_mask.sum()), 'Total_Obs_Artifact': dropped_anchor
        })
    df_clean = df_clean[anchor_mask].copy()

    # --- Step 3: Delete rows where ALL obs variables are NaN ---
    obs_cols = [c for c, cl in cols_lower.items() if cl not in CORE_ANCHOR_VARS]
    if obs_cols:
        has_any_obs = df_clean[obs_cols].notna().any(axis=1)
        dropped_empty = int((~has_any_obs).sum())
        if dropped_empty > 0:
            logs.append({
                'Group': group_name, 'Variable': 'ALL_OBS',
                'Issue_Type': f'Row Deleted — all obs NaN ({dropped_empty} rows)',
                'Bad_Obs_Valid': 0, 'Bad_Obs_Artifact': dropped_empty,
                'Total_Obs_Valid': int(has_any_obs.sum()), 'Total_Obs_Artifact': dropped_empty
            })
        df_clean = df_clean[has_any_obs].copy()

    # --- Step 4: Delete group if any location column is entirely NaN ---
    #
    # If a spatial locator column (lat, lon, height, altitude, ght) exists
    # but every value in it is NaN, the group has no usable spatial context
    # and must not be written.  In practice lat/lon cannot reach this state
    # (Step 2 already deletes rows with NaN anchors), so this fires only for
    # vertical coordinates that were fully wiped out by bounds QC in Step 1
    # while horizontal position and time remained valid.
    # Must stay in sync with LOCATION_VARS in hrdobs_v1.0_validate_ai_ready_batch.py.
    _LOCATION_VARS = {'lat', 'latitude', 'lon', 'longitude', 'height', 'altitude', 'ght'}
    for col, col_l in cols_lower.items():
        if col_l not in _LOCATION_VARS:
            continue
        if col not in df_clean.columns:
            continue
        if df_clean[col].isna().all():
            n_dropped = len(df_clean)
            logs.append({
                'Group': group_name, 'Variable': col,
                'Issue_Type': (
                    f'Group Deleted — location column "{col}" entirely NaN '
                    f'after bounds QC ({n_dropped} rows dropped)'
                ),
                'Bad_Obs_Valid': 0, 'Bad_Obs_Artifact': n_dropped,
                'Total_Obs_Valid': 0, 'Total_Obs_Artifact': n_dropped,
            })
            df_clean = df_clean.iloc[0:0].copy()
            break   # one all-NaN location column is enough to discard the group

    return df_clean, logs


# =============================================================================
# CORE CONVERSION
# =============================================================================

def convert_universal(input_h5, output_h5, scan_only=False, spline_alt_mb=None,
                      ships_lookup=None, error_sim_logs=None):
    """
    Convert one original HRDOBS HDF5 file to the AI-ready flat format.

    Parameters
    ----------
    spline_alt_mb : float or None
        If provided, a constant 'pres' dataset (in hPa) is written into
        the track_spline_track group, representing the flight-level
        pressure at which the spline track center fixes were obtained.
    ships_lookup : dict or None
        Lookup table returned by load_ships_lookup().  When provided and
        a matching SHIPS cycle is found, a 'ships_params' group containing
        the 25 t=0 SHIPS predictors is written into the output file.  Any
        discrepancies between the SHIPS HEAD metadata and the HRDOBS global
        metadata are logged to qc_logs with an 'SHIPS_MISMATCH:' prefix.

    Returns (inventory_entry, schema_updates, qc_logs).
    """
    if error_sim_logs is None:
        error_sim_logs = []
    qc_logs          = []
    ships_mismatches = []   # structured per-field mismatch records for dedicated CSV
    ships_no_matches  = []   # structured no-match records for dedicated CSV
    schema_updates = {'global': set(), 'groups': {}}
    inventory_entry = None

    try:
        null_ctx = open(os.devnull, 'w')
        out_ctx  = h5py.File(output_h5, 'w') if not scan_only else null_ctx

        with h5py.File(input_h5, 'r') as f_in, out_ctx as f_out:

            global_attrs = {}

            # ----------------------------------------------------------------
            # Helper: accumulate global attributes into dict
            # ----------------------------------------------------------------
            # Attributes that are internal/debug artifacts in some source files
            # and should never appear in the published AI-ready files.
            EXCLUDED_GLOBAL_ATTRS = {
                'manifest expected', 'original source count', 'validation status', 'platforms'
            }
            
            def set_meta(k, v):
                if k.lower() in EXCLUDED_GLOBAL_ATTRS:
                    return
                # Skip attributes that are always computed by the conversion
                # (never passed through from source).
                if k.lower() in {a.lower() for a in CONTROLLED_ATTRS}:
                    return
                global_attrs[k] = decode_attr(v)

            # ----------------------------------------------------------------
            # 1. Global metadata
            # ----------------------------------------------------------------
            for k, v in extract_filename_metadata(input_h5).items():
                set_meta(k, v)
            for k, v in f_in.attrs.items():
                set_meta(k, v)
            if 'storm stats' in f_in:
                for k, v in f_in['storm stats'].attrs.items():
                    if k.lower() != 'center_from_best_track':
                        set_meta(k, v)

            if 'center_from_tc_vitals' not in global_attrs:
                set_meta('center_from_tc_vitals', 'NaN')
            if 'tc_category' not in global_attrs:
                set_meta('tc_category', 'NaN')

            # Capture storm_motion from source (set_meta skips it as
            # CONTROLLED, so we read it directly for later use).
            source_storm_motion = 'NaN'
            if 'storm stats' in f_in:
                sm = f_in['storm stats'].attrs.get('storm_motion', None)
                if sm is not None:
                    source_storm_motion = decode_attr(sm)
                    if not source_storm_motion.strip():
                        source_storm_motion = 'NaN'
            elif 'storm_motion' in f_in.attrs:
                sm = f_in.attrs['storm_motion']
                source_storm_motion = decode_attr(sm)
                if not source_storm_motion.strip():
                    source_storm_motion = 'NaN'

            # Capture source geospatial bounds for cross-check later.
            source_geo = {}
            for geo_key in ('geospatial_lat_max', 'geospatial_lat_min',
                            'geospatial_lon_max', 'geospatial_lon_min'):
                gv = f_in.attrs.get(geo_key, None)
                if gv is not None:
                    try:
                        source_geo[geo_key] = float(decode_attr(gv))
                    except (ValueError, TypeError):
                        pass

            # ----------------------------------------------------------------
            # 2. Best track intensity & center at file valid time
            # ----------------------------------------------------------------
            intensity_val, mslp_val, rmw_val = "NaN", "NaN", "NaN"
            clat_val, clon_val = None, None
            
            match = re.search(r'[\._](\d{4})(\d{2})(\d{2})(\d{2})', input_h5)

            # Extract cycle_dt for time QC (Layers 1-3).
            # Available to process_obs_leaf via closure and passed explicitly
            # to validate_and_clean_data and track group QC below.
            cycle_dt = None
            if match:
                try:
                    cycle_dt = datetime.datetime(
                        int(match.group(1)), int(match.group(2)),
                        int(match.group(3)), int(match.group(4)),
                        0, 0, tzinfo=datetime.timezone.utc
                    )
                except (ValueError, OverflowError):
                    pass

            if match and 'track' in f_in and 'best track' in f_in['track']:
                target_time_str = f"{match.group(1)}{match.group(2)}{match.group(3)}{match.group(4)}0000"
                target_dt = pd.to_datetime(target_time_str, format="%Y%m%d%H%M%S")
                
                bt = f_in['track']['best track']
                if 'time' in bt:
                    times = bt['time'][:]

                    if len(times) == 0:
                        pass   # empty best track — leave intensity as NaN
                    else:
                        try:
                            bt_dts = pd.to_datetime(times, format='%Y%m%d%H%M%S',
                                                    errors='coerce')
                            # Drop any entries that failed to parse (NaT)
                            valid_mask    = bt_dts.notna()
                            bt_dts_valid  = bt_dts[valid_mask]
                            times_valid   = times[valid_mask]

                            if len(bt_dts_valid) == 0:
                                raise ValueError("No parseable time values in best track")

                            time_diffs_sec = np.abs(
                                (bt_dts_valid - target_dt).total_seconds()
                            ).to_numpy()
                            best_idx = int(np.argmin(time_diffs_sec))

                            if time_diffs_sec[best_idx] <= 10800:
                                i = best_idx
                                # Correct lowercase field names
                                if 'vmax' in bt:
                                    intensity_val = float(bt['vmax'][i])
                                if 'pmin' in bt:
                                    raw_p = float(bt['pmin'][i])
                                    mslp_val = raw_p / 100.0 if raw_p > 2000 else raw_p
                                if 'rmw' in bt:
                                    rmw_val = float(bt['rmw'][i])
                                if 'clat' in bt: clat_val = float(bt['clat'][i])
                                elif 'lat' in bt: clat_val = float(bt['lat'][i])
                                if 'clon' in bt: clon_val = float(bt['clon'][i])
                                elif 'lon' in bt: clon_val = float(bt['lon'][i])
                                if clon_val is not None and clon_val > 0:
                                    clon_val = -clon_val
                            else:
                                qc_logs.append({
                                    'Filename': os.path.basename(input_h5),
                                    'Group': 'track_best_track', 'Variable': 'time',
                                    'Issue_Type': (
                                        f'WARNING: No best track point found within 3 hours of '
                                        f'{target_time_str} — intensity metadata set to NaN'
                                    ),
                                    'Bad_Obs_Valid': 0, 'Bad_Obs_Artifact': 0,
                                    'Total_Obs_Valid': len(times), 'Total_Obs_Artifact': 0
                                })

                        except Exception as bt_err:
                            qc_logs.append({
                                'Filename': os.path.basename(input_h5),
                                'Group': 'track_best_track', 'Variable': 'time',
                                'Issue_Type': (
                                    f'WARNING: Could not parse best track times '
                                    f'({bt_err}) — intensity metadata set to NaN'
                                ),
                                'Bad_Obs_Valid': 0, 'Bad_Obs_Artifact': 0,
                                'Total_Obs_Valid': len(times), 'Total_Obs_Artifact': 0
                            })

            set_meta('storm_intensity_ms', intensity_val)
            set_meta('storm_mslp_hpa',     mslp_val)
            set_meta('radius_of_maximum_wind_km', rmw_val)

            # ----------------------------------------------------------------
            # Resolve tc_category to a standardized token.
            # ----------------------------------------------------------------
            raw_cat_val = "NaN"
            if 'storm stats' in f_in:
                raw_cat_val = decode_attr(
                    f_in['storm stats'].attrs.get('tc_category', 'NaN')
                )

            resolved_cat, cat_source = resolve_tc_category(raw_cat_val, intensity_val)
            set_meta('tc_category', resolved_cat)

            if cat_source == 'derived':
                qc_logs.append({
                    'Filename':   os.path.basename(input_h5),
                    'Group':      'storm_stats',
                    'Variable':   'tc_category',
                    'Issue_Type': (
                        f'Advisory: tc_category derived from intensity '
                        f'(raw="{raw_cat_val}", intensity={intensity_val} m/s '
                        f'→ resolved="{resolved_cat}")'
                    ),
                    'Bad_Obs_Valid': 0, 'Bad_Obs_Artifact': 0,
                    'Total_Obs_Valid': 0, 'Total_Obs_Artifact': 0,
                })
            elif cat_source == 'unknown':
                qc_logs.append({
                    'Filename':   os.path.basename(input_h5),
                    'Group':      'storm_stats',
                    'Variable':   'tc_category',
                    'Issue_Type': (
                        f'WARNING: tc_category missing and intensity unavailable '
                        f'— category set to NaN. '
                        f'raw tc_category="{raw_cat_val}", '
                        f'intensity_ms="{intensity_val}"'
                    ),
                    'Bad_Obs_Valid': 0, 'Bad_Obs_Artifact': 0,
                    'Total_Obs_Valid': 0, 'Total_Obs_Artifact': 0,
                })
            
            if clat_val is not None and clon_val is not None:
                set_meta('center_from_tc_vitals', f"{clat_val}, {clon_val}")
            else:
                if 'center_from_tc_vitals' not in global_attrs:
                    set_meta('center_from_tc_vitals', 'NaN')

            # ----------------------------------------------------------------
            # 2b. Metadata completeness gate
            # ----------------------------------------------------------------
            if not scan_only:
                nan_critical = []
                for crit_key in sorted(CRITICAL_METADATA):
                    # Check pre-processed global_attrs to safely assess completeness
                    crit_val = str(global_attrs.get(crit_key, 'NaN')).strip()
                    if crit_val in ('NaN', 'nan', ''):
                        nan_critical.append(crit_key)

                if nan_critical:
                    qc_logs.append({
                        'Filename': os.path.basename(input_h5),
                        'Group':    'METADATA',
                        'Variable': ', '.join(nan_critical),
                        'Issue_Type': (
                            f'SKIP: Incomplete critical metadata — '
                            f'{len(nan_critical)} field(s) are NaN: '
                            f'{", ".join(nan_critical)}. '
                            f'File will not be converted.'
                        ),
                        'Bad_Obs_Valid': 0, 'Bad_Obs_Artifact': 0,
                        'Total_Obs_Valid': 0, 'Total_Obs_Artifact': 0,
                    })
                    f_out.close()
                    try:
                        os.remove(output_h5)
                    except OSError:
                        pass
                    return None, schema_updates, qc_logs, ships_mismatches, ships_no_matches

            # ----------------------------------------------------------------
            # 3. Observation groups (flat: instrument_type_platform)
            # ----------------------------------------------------------------
            obs_groups  = []
            group_counts = {}
            all_vars    = set()

            def process_obs_leaf(group_name, sub_grp, parent_missing=None):
                """Read, QC, and write one flat observation group."""
                grp_missing = sub_grp.attrs.get('missing_value', parent_missing)

                # --- Read datasets + replace missing values ---
                df_dict  = {}
                for dset_name, dset in sub_grp.items():
                    if not isinstance(dset, h5py.Dataset):
                        continue
                    raw = dset[:]
                    if np.issubdtype(raw.dtype, np.number):
                        raw = replace_missing_values(raw, grp_missing)
                    df_dict[dset_name] = raw

                if not df_dict:
                    return

                # --- Assemble DataFrame ---
                df = pd.DataFrame(df_dict)

                # --- Standard QC (bounds + anchor + all-NaN row checks) ---
                df_clean, logs = validate_and_clean_data(df, group_name,
                                                          cycle_dt=cycle_dt)
                for log in logs:
                    log['Filename'] = os.path.basename(input_h5)
                    qc_logs.append(log)

                # Storm-center proximity filter
                if clat_val is not None and clon_val is not None and len(df_clean) > 0:
                    cl = {c.lower(): c for c in df_clean.columns}
                    lat_col = next((cl[k] for k in ['lat', 'latitude'] if k in cl), None)
                    lon_col = next((cl[k] for k in ['lon', 'longitude'] if k in cl), None)
                    if lat_col and lon_col:
                        prox_mask = (
                            (df_clean[lat_col] >= clat_val - OBS_PROXIMITY_DEG) &
                            (df_clean[lat_col] <= clat_val + OBS_PROXIMITY_DEG) &
                            (df_clean[lon_col] >= clon_val - OBS_PROXIMITY_DEG) &
                            (df_clean[lon_col] <= clon_val + OBS_PROXIMITY_DEG)
                        )
                        dropped_prox = int((~prox_mask).sum())
                        if dropped_prox > 0:
                            qc_logs.append({
                                'Filename': os.path.basename(input_h5),
                                'Group': group_name, 'Variable': 'lat/lon',
                                'Issue_Type': (
                                    f'Proximity Filter — {dropped_prox} row(s) deleted: '
                                    f'outside ±{OBS_PROXIMITY_DEG}° of storm center '
                                    f'({clat_val:.2f}N, {clon_val:.2f}E)'
                                ),
                                'Bad_Obs_Valid': 0, 'Bad_Obs_Artifact': dropped_prox,
                                'Total_Obs_Valid': int(prox_mask.sum()),
                                'Total_Obs_Artifact': dropped_prox,
                            })
                            df_clean = df_clean[prox_mask].copy()

                # --- Time Layer 3: minimum span advisory ---
                _check_time_span(df_clean, group_name,
                                 os.path.basename(input_h5), qc_logs)

                if len(df_clean) == 0:
                    qc_logs.append({
                        'Filename': os.path.basename(input_h5),
                        'Group': group_name, 'Variable': 'ALL',
                        'Issue_Type': 'STRUCTURAL: Group deleted — 100% garbage rows',
                        'Bad_Obs_Valid': len(df), 'Bad_Obs_Artifact': 0,
                        'Total_Obs_Valid': 0, 'Total_Obs_Artifact': len(df)
                    })
                    return

                # Assign observation errors per OBS_ERROR_CONFIG rules.
                rules = OBS_ERROR_CONFIG.get(group_name, {})
                assigned_errors = set()
                
                for col in list(df_clean.columns):
                    if col.endswith('err') or col in COORD_VARS or col in TRACK_PRODUCT_VARS:
                        continue
                        
                    base_var = col
                    err_name = f"{base_var}err"
                    
                    base_valid_mask = df_clean[base_var].notna()
                    base_valid_count = int(base_valid_mask.sum())
                    
                    if base_valid_count == 0:
                        continue
                        
                    rule = rules.get(err_name)
                    if not rule:
                        fallback_val = ERROR_FALLBACK_RULES.get(base_var)
                        if fallback_val is not None:
                             rule = {'type': 'DYNAMIC_FALLBACK', 'value': fallback_val}
                        else:
                             continue

                    rule_type = rule['type']
                    rule_val = float(rule['value'])
                    
                    has_original = err_name in df_clean.columns
                    original_valid_count = 0
                    if has_original:
                        original_valid_count = int((df_clean[err_name].notna() & base_valid_mask).sum())
                    else:
                        df_clean[err_name] = np.nan
                        
                    if rule_type == 'CONSTANT':
                        injected = base_valid_count
                        if scan_only:
                            if not has_original: action = "ACTION_INJECTED"
                            elif original_valid_count > 0: action = "ACTION_OVERWRITTEN"
                            else: action = "ACTION_INJECTED"
                            
                            error_sim_logs.append({
                                'Filename': os.path.basename(input_h5),
                                'Group': group_name,
                                'Variable': err_name,
                                'Action_Type': action,
                                'Base_Valid_Count': base_valid_count,
                                'Original_Valid_Count': original_valid_count,
                                'Simulated_Injections': injected
                            })
                            
                        df_clean[err_name] = np.nan
                        df_clean.loc[base_valid_mask, err_name] = rule_val
                        assigned_errors.add(err_name)
                        
                    elif rule_type == 'DYNAMIC_FALLBACK':
                        if has_original:
                            missing_mask = df_clean[err_name].isna() & base_valid_mask
                        else:
                            missing_mask = base_valid_mask
                            
                        injected = int(missing_mask.sum())
                        
                        if scan_only and injected > 0:
                            error_sim_logs.append({
                                'Filename': os.path.basename(input_h5),
                                'Group': group_name,
                                'Variable': err_name,
                                'Action_Type': 'ACTION_FALLBACK_APPLIED',
                                'Base_Valid_Count': base_valid_count,
                                'Original_Valid_Count': original_valid_count,
                                'Simulated_Injections': injected
                            })
                            
                        df_clean.loc[missing_mask, err_name] = rule_val
                        assigned_errors.add(err_name)

                # --- NaN-sync: enforce bidirectional NaN parity ---
                for col in df_clean.columns:
                    if not col.endswith('err'):
                        continue
                    base_var = col[:-3]
                    if base_var not in df_clean.columns:
                        continue
                    # Direction 1 (both types): base NaN -> err NaN
                    df_clean.loc[df_clean[base_var].isna(), col] = np.nan
                    # Direction 2 (native arrays only): err NaN -> base NaN
                    if col not in assigned_errors:
                        df_clean.loc[df_clean[col].isna(), base_var] = np.nan

                obs_groups.append(group_name)
                group_counts[group_name] = len(df_clean)

                if group_name not in schema_updates['groups']:
                    schema_updates['groups'][group_name] = {
                        'attrs': set(), 'datasets': set()
                    }

                if not scan_only:
                    out_grp = f_out.create_group(group_name)
                    out_grp.attrs['obs_count'] = len(df_clean)

                    for ak, av in sub_grp.attrs.items():
                        out_grp.attrs[ak] = safe_attr(av)
                        schema_updates['groups'][group_name]['attrs'].add(ak)

                    for col in df_clean.columns:
                        data_arr = df_clean[col].values.astype(np.float64)
                        
                        if col == 'time':
                            data_arr = convert_packed_time_to_cf(data_arr)

                        dset_out = out_grp.create_dataset(col, data=data_arr)
                        schema_updates['groups'][group_name]['datasets'].add(col)
                        all_vars.add(col)

                        dset_out.attrs['fill_value'] = np.nan

                        if col in sub_grp:
                            for ak, av in sub_grp[col].attrs.items():
                                dset_out.attrs[ak] = safe_attr(av)
                            if col.endswith('err'):
                                base_var = col[:-3]
                                dset_out.attrs['long_name'] = (
                                    f"error estimate for {base_var}"
                                )
                        elif col in assigned_errors:
                            base_var = col[:-3]
                            dset_out.attrs['long_name'] = (
                                f"error estimate for {base_var}"
                            )
                            if (base_var in sub_grp and
                                    'units' in sub_grp[base_var].attrs):
                                dset_out.attrs['units'] = safe_attr(
                                    sub_grp[base_var].attrs['units']
                                )
                                
                        if col == 'time':
                            dset_out.attrs['units'] = 'seconds since 1900-01-01 00:00:00Z'
                            dset_out.attrs['calendar'] = 'gregorian'
                            dset_out.attrs['standard_name'] = 'time'
                            dset_out.attrs['long_name'] = 'Time of observation'

            # --- Main obs traversal ---
            for top_key in f_in.keys():
                top_key_l = top_key.lower()
                if top_key_l in ('track', 'storm stats'):
                    continue
                if any(b in top_key_l for b in BANNED_INSTRUMENTS):
                    continue

                for mid_key in f_in[top_key].keys():
                    mid_grp   = f_in[top_key][mid_key]
                    mid_key_l = mid_key.lower()
                    if any(b in mid_key_l for b in BANNED_INSTRUMENTS):
                        continue

                    # Skip if mid_grp is itself a Dataset (shouldn't happen, but guard it)
                    if not isinstance(mid_grp, h5py.Group):
                        continue

                    grp_missing_mid = mid_grp.attrs.get('missing_value', None)

                    # Detect depth: peek at the first child of mid_grp
                    first_child = next(iter(mid_grp.values()), None)

                    if first_child is None:
                        continue
                    elif isinstance(first_child, h5py.Dataset):
                        # 2-level: mid_grp IS the leaf — flat name is just mid_key
                        group_name = mid_key.replace(" ", "_")
                        process_obs_leaf(group_name, mid_grp, grp_missing_mid)
                    else:
                        # 3-level: iterate platforms inside mid_grp
                        for plat_key in mid_grp.keys():
                            plat_grp = mid_grp[plat_key]
                            if not isinstance(plat_grp, h5py.Group):
                                continue
                            group_name = f"{mid_key}_{plat_key}".replace(" ", "_")
                            process_obs_leaf(group_name, plat_grp, grp_missing_mid)

            # ----------------------------------------------------------------
            # 4. Track groups (sorted by time, bounds-cleaned)
            #    Track groups are intentionally exempt from the proximity
            #    filter — they represent the full storm lifecycle and will
            #    naturally extend well beyond the ±OBS_PROXIMITY_DEG box
            #    of any single cycle.
            # ----------------------------------------------------------------
            if 'track' in f_in:
                for sub_key in f_in['track'].keys():
                    group_name = f"track_{sub_key}".replace(" ", "_")
                    sub_grp    = f_in['track'][sub_key]

                    df_dict = {}
                    grp_missing = sub_grp.attrs.get('missing_value', None)

                    for dset_name, dset in sub_grp.items():
                        if not isinstance(dset, h5py.Dataset):
                            continue
                        raw = dset[:]
                        if np.issubdtype(raw.dtype, np.number):
                            raw = replace_missing_values(raw, grp_missing)
                        df_dict[dset_name] = raw

                    if not df_dict:
                        continue

                    df = pd.DataFrame(df_dict)

                    if len(df) == 0:
                        qc_logs.append({
                            'Filename':   os.path.basename(input_h5),
                            'Group':      group_name,
                            'Variable':   'ALL',
                            'Issue_Type': (
                                f'WARNING: Track group "{group_name}" is empty '
                                f'(0 rows after missing-value replacement) — '
                                f'file should be excluded from the dataset.'
                            ),
                            'Bad_Obs_Valid': 0, 'Bad_Obs_Artifact': 0,
                            'Total_Obs_Valid': 0, 'Total_Obs_Artifact': 0,
                        })
                        continue

                    # Flip positive longitudes to negative (West convention).
                    for lon_col in [c for c in df.columns if c.lower() in ('lon', 'clon')]:
                        df[lon_col] = df[lon_col].apply(
                            lambda x: -x if (pd.notna(x) and x > 0) else x
                        )

                    # Bounds QC for track group variables — replace out-of-
                    # bounds values with NaN and log to qc_logs.
                    for col in df.columns:
                        col_l = col.lower()
                        if col_l in VALID_BOUNDS:
                            lo, hi = VALID_BOUNDS[col_l]
                            bad_mask = (df[col] < lo) | (df[col] > hi)
                            bad_mask = bad_mask & df[col].notna()
                            if bad_mask.any():
                                bad_vals = df.loc[bad_mask, col].tolist()
                                df.loc[bad_mask, col] = np.nan
                                qc_logs.append({
                                    'Filename':  os.path.basename(input_h5),
                                    'Group':     group_name,
                                    'Variable':  col,
                                    'Issue_Type': (
                                        f'Track Bounds QC — {len(bad_vals)} '
                                        f'value(s) outside [{lo}, {hi}] '
                                        f'replaced with NaN: '
                                        f'{[round(v,2) for v in bad_vals[:10]]}'
                                        + (f' ...(+{len(bad_vals)-10} more)'
                                           if len(bad_vals) > 10 else '')
                                    ),
                                    'Bad_Obs_Valid':    len(bad_vals),
                                    'Bad_Obs_Artifact': 0,
                                    'Total_Obs_Valid':  len(df),
                                    'Total_Obs_Artifact': 0,
                                })

                    # Time QC Layers 1-2 for track groups.
                    # Track groups use ±6h (wider than obs groups) to
                    # accommodate bracketing best track fixes at ±6h.
                    if 'time' in df.columns:
                        track_time_logs = []
                        _apply_time_qc(df, 'time', group_name, cycle_dt,
                                       window_hours=6, logs=track_time_logs)
                        for log in track_time_logs:
                            log['Filename'] = os.path.basename(input_h5)
                            qc_logs.append(log)

                    # After bounds QC: if time is entirely NaN, the track is
                    # unusable — log and skip.
                    if 'time' in df.columns:
                        if df['time'].isna().all():
                            qc_logs.append({
                                'Filename':  os.path.basename(input_h5),
                                'Group':     group_name,
                                'Variable':  'time',
                                'Issue_Type': (
                                    f'WARNING: Track group "{group_name}" has '
                                    f'no valid time values after bounds QC — '
                                    f'group skipped.'
                                ),
                                'Bad_Obs_Valid': 0, 'Bad_Obs_Artifact': len(df),
                                'Total_Obs_Valid': 0, 'Total_Obs_Artifact': len(df),
                            })
                            continue

                    # Sort by time to ensure monotonic ordering.
                    if 'time' in df.columns:
                        df = df.sort_values('time', na_position='last') \
                               .reset_index(drop=True)

                    # Time Layer 3: minimum span advisory for track groups.
                    _check_time_span(df, group_name,
                                     os.path.basename(input_h5), qc_logs)

                    obs_groups.append(group_name)
                    group_counts[group_name] = len(df)

                    if group_name not in schema_updates['groups']:
                        schema_updates['groups'][group_name] = {
                            'attrs': set(), 'datasets': set()
                        }

                    if not scan_only:
                        out_grp = f_out.create_group(group_name)
                        out_grp.attrs['obs_count'] = len(df)

                        for ak, av in sub_grp.attrs.items():
                            out_grp.attrs[ak] = safe_attr(av)
                            schema_updates['groups'][group_name]['attrs'].add(ak)

                        for col in df.columns:
                            data_arr = df[col].values.astype(np.float64)

                            # Skip datasets that are entirely NaN after
                            # missing-value replacement.
                            if np.all(np.isnan(data_arr)):
                                continue
                                
                            if col == 'time':
                                data_arr = convert_packed_time_to_cf(data_arr)

                            dset_out = out_grp.create_dataset(col, data=data_arr)
                            schema_updates['groups'][group_name]['datasets'].add(col)
                            all_vars.add(col)
                            dset_out.attrs['fill_value'] = np.nan
                            
                            if col in sub_grp:
                                for ak, av in sub_grp[col].attrs.items():
                                    dset_out.attrs[ak] = safe_attr(av)
                                    
                            if col == 'time':
                                dset_out.attrs['units'] = 'seconds since 1900-01-01 00:00:00Z'
                                dset_out.attrs['calendar'] = 'gregorian'
                                dset_out.attrs['standard_name'] = 'time'
                                dset_out.attrs['long_name'] = 'Time of observation'

                        # ── Inject flight-level pressure into spline track ──
                        if (group_name == 'track_spline_track' and
                                spline_alt_mb is not None and
                                len(df) > 0):
                            pres_arr = np.full(len(df), float(spline_alt_mb),
                                               dtype=np.float64)
                            pres_dset = out_grp.create_dataset('pres', data=pres_arr)
                            pres_dset.attrs['fill_value'] = np.nan
                            pres_dset.attrs['units'] = 'hPa'
                            pres_dset.attrs['long_name'] = (
                                'flight-level pressure of aircraft center fixes'
                            )
                            schema_updates['groups'][group_name]['datasets'].add('pres')
                            all_vars.add('pres')

            # ----------------------------------------------------------------
            # 4b. Write controlled metadata attributes (via process dict)
            # ----------------------------------------------------------------
            
            # ── Constants ──────────────────────────────────────────
            global_attrs['creator_email']  = CREATOR_EMAIL
            global_attrs['creator_name']   = CREATOR_NAME
            global_attrs['title']          = TITLE
            global_attrs['version_number'] = VERSION_NUMBER

            # ── Groups (Derived and Expected) ─────────────────────
            global_attrs['existing_groups'] = sorted(obs_groups)
            global_attrs['expected_groups'] = EXPECTED_GROUPS_LIST

            global_attrs['Virtual_Manifest'] = os.path.basename(output_h5).replace('.hdf5', '.json') if output_h5 else 'NaN'

            # ── Storm motion (from source, or NaN) ────────────────
            global_attrs['storm_motion'] = source_storm_motion

            # ── Time coverage (from cycle time ± 3h) ──────────────
            fn_match = re.search(
                r'[\._](\d{4})(\d{2})(\d{2})(\d{2})(\d{2})?',
                input_h5
            )
            if fn_match:
                yr  = int(fn_match.group(1))
                mo  = int(fn_match.group(2))
                dy  = int(fn_match.group(3))
                hr  = int(fn_match.group(4))
                mn  = int(fn_match.group(5)) if fn_match.group(5) else 0
                cycle_dt = datetime.datetime(yr, mo, dy, hr, mn, 0,
                                            tzinfo=datetime.timezone.utc)
                tc_start = cycle_dt - datetime.timedelta(hours=3)
                tc_end   = cycle_dt + datetime.timedelta(hours=3)
                global_attrs['time_coverage_start'] = tc_start.strftime("%Y-%m-%dT%H:%M:%SZ")
                global_attrs['time_coverage_end']   = tc_end.strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                global_attrs['time_coverage_start'] = 'NaN'
                global_attrs['time_coverage_end']   = 'NaN'

            # ── Geospatial bounding box ───────────────────────────
            if clat_val is not None and clon_val is not None:
                geo_lat_min = clat_val - GEOSPATIAL_HALFWIDTH
                geo_lat_max = clat_val + GEOSPATIAL_HALFWIDTH
                geo_lon_min = clon_val - GEOSPATIAL_HALFWIDTH
                geo_lon_max = clon_val + GEOSPATIAL_HALFWIDTH

                # Cross-check against source bounds if available
                if source_geo:
                    for geo_key, computed_val in [
                        ('geospatial_lat_min', geo_lat_min),
                        ('geospatial_lat_max', geo_lat_max),
                        ('geospatial_lon_min', geo_lon_min),
                        ('geospatial_lon_max', geo_lon_max),
                    ]:
                        if geo_key in source_geo:
                            diff = abs(source_geo[geo_key] - computed_val)
                            if diff > 1.0:
                                qc_logs.append({
                                    'Filename': os.path.basename(input_h5),
                                    'Group':    'METADATA',
                                    'Variable': geo_key,
                                    'Issue_Type': (
                                        f'Geospatial bounds mismatch: '
                                        f'source={source_geo[geo_key]:.2f}, '
                                        f'computed from center='
                                        f'{computed_val:.2f} '
                                        f'(diff={diff:.2f}°)'
                                    ),
                                    'Bad_Obs_Valid': 0,
                                    'Bad_Obs_Artifact': 0,
                                    'Total_Obs_Valid': 0,
                                    'Total_Obs_Artifact': 0,
                                })

                global_attrs['geospatial_lat_min'] = round(geo_lat_min, 2)
                global_attrs['geospatial_lat_max'] = round(geo_lat_max, 2)
                global_attrs['geospatial_lon_min'] = round(geo_lon_min, 2)
                global_attrs['geospatial_lon_max'] = round(geo_lon_max, 2)
            else:
                # No center available — write NaN
                global_attrs['geospatial_lat_min'] = np.nan
                global_attrs['geospatial_lat_max'] = np.nan
                global_attrs['geospatial_lon_min'] = np.nan
                global_attrs['geospatial_lon_max'] = np.nan

            global_attrs['geospatial_lat_units'] = 'degrees north'
            global_attrs['geospatial_lon_units'] = 'degrees east'

            # ── Write all global metadata ──────────────────────────
            cleaned_attrs = process_root_metadata(global_attrs)
            for k, v in cleaned_attrs.items():
                if not scan_only:
                    f_out.attrs[k] = v
                schema_updates['global'].add(k)


            # ----------------------------------------------------------------
            # 4c. SHIPS parameters group
            # ----------------------------------------------------------------
            if ships_lookup:
                _file_meta  = extract_filename_metadata(input_h5)
                _storm_id   = _file_meta.get('storm_id', '')
                _storm_dt   = _file_meta.get('storm_datetime', '')
                _ships_key  = _ships_key_from_hrdobs(_storm_id, _storm_dt)
                _ships_row  = ships_lookup.get(_ships_key) if _ships_key else None

                if _ships_row is not None:
                    # ── Write ships_params group (write mode only) ────────
                    if not scan_only:
                        sp_grp = f_out.create_group('ships_params')
                        sp_grp.attrs['obs_count']          = 1
                        sp_grp.attrs['source']             = 'SHIPS lsdiag predictor file'
                        sp_grp.attrs['ships_atcf_id']      = _ships_row.get('atcf_id', '')
                        sp_grp.attrs['ships_datetime_utc'] = _ships_row.get('datetime_utc', '')

                        obs_groups.append('ships_params')
                        group_counts['ships_params'] = 1
                        schema_updates['groups'].setdefault('ships_params', {
                            'attrs': set(), 'datasets': set()
                        })
                        schema_updates['groups']['ships_params']['attrs'].update({
                            'obs_count', 'source', 'ships_atcf_id', 'ships_datetime_utc'
                        })

                        for col, (units, long_name) in SHIPS_PREDICTOR_META.items():
                            val_str = _ships_row.get(col, '')
                            try:
                                val = float(val_str) if val_str != '' else np.nan
                            except (ValueError, TypeError):
                                val = np.nan

                            data_arr = np.array([val], dtype=np.float64)
                            dset     = sp_grp.create_dataset(col, data=data_arr)
                            dset.attrs['fill_value'] = np.nan
                            dset.attrs['units']      = units
                            dset.attrs['long_name']  = long_name
                            schema_updates['groups']['ships_params']['datasets'].add(col)
                            all_vars.add(col)

                    # ── Metadata mismatch checking (scan + write modes) ───
                    # Each entry: (field_name, hrdobs_val_str, ships_val_str, diff, units, tol)
                    _mm = []

                    # Center latitude
                    try:
                        s_lat = float(_ships_row.get('lat_hd_degN', ''))
                        if clat_val is not None and not np.isnan(s_lat):
                            d = abs(clat_val - s_lat)
                            if d > SHIPS_MISMATCH_TOL_LAT:
                                _mm.append(('center_lat',
                                            f'{clat_val:.4f}', f'{s_lat:.4f}',
                                            d, 'deg_N', SHIPS_MISMATCH_TOL_LAT))
                    except (ValueError, TypeError):
                        pass

                    # Center longitude
                    try:
                        s_lon = float(_ships_row.get('lon_hd_degE', ''))
                        if clon_val is not None and not np.isnan(s_lon):
                            d = abs(clon_val - s_lon)
                            if d > SHIPS_MISMATCH_TOL_LON:
                                _mm.append(('center_lon',
                                            f'{clon_val:.4f}', f'{s_lon:.4f}',
                                            d, 'deg_E', SHIPS_MISMATCH_TOL_LON))
                    except (ValueError, TypeError):
                        pass

                    # Intensity — HRDOBS in m/s, SHIPS in kt → convert for comparison
                    try:
                        s_vmax_kt = float(_ships_row.get('vmax_hd_kt', ''))
                        h_vmax_ms = float(intensity_val)
                        if not np.isnan(s_vmax_kt) and not np.isnan(h_vmax_ms):
                            h_vmax_kt = h_vmax_ms * 1.94384
                            d = abs(h_vmax_kt - s_vmax_kt)
                            if d > SHIPS_MISMATCH_TOL_VMAX:
                                _mm.append(('vmax',
                                            f'{h_vmax_kt:.2f}', f'{s_vmax_kt:.2f}',
                                            d, 'kt', SHIPS_MISMATCH_TOL_VMAX))
                    except (ValueError, TypeError):
                        pass

                    # MSLP
                    try:
                        s_mslp = float(_ships_row.get('mslp_hd_hpa', ''))
                        h_mslp = float(mslp_val)
                        if not np.isnan(s_mslp) and not np.isnan(h_mslp):
                            d = abs(h_mslp - s_mslp)
                            if d > SHIPS_MISMATCH_TOL_MSLP:
                                _mm.append(('mslp',
                                            f'{h_mslp:.2f}', f'{s_mslp:.2f}',
                                            d, 'hPa', SHIPS_MISMATCH_TOL_MSLP))
                    except (ValueError, TypeError):
                        pass

                    for field, hrdobs_v, ships_v, diff, units, tol in _mm:
                        # Forensics report entry (human-readable summary)
                        qc_logs.append({
                            'Filename':       os.path.basename(input_h5),
                            'Group':          'ships_params',
                            'Variable':       'METADATA_MISMATCH',
                            'Issue_Type':     (
                                f'SHIPS_MISMATCH: {field}: '
                                f'HRDOBS={hrdobs_v}, SHIPS={ships_v}, '
                                f'diff={diff:.3g} {units}'
                            ),
                            'Bad_Obs_Valid':   0, 'Bad_Obs_Artifact': 0,
                            'Total_Obs_Valid': 0, 'Total_Obs_Artifact': 0,
                        })
                        # Dedicated mismatch CSV entry (structured)
                        ships_mismatches.append({
                            'Filename':      os.path.basename(input_h5),
                            'Storm_ID':      _storm_id,
                            'Datetime_UTC':  _storm_dt,
                            'SHIPS_ATCF_ID': _ships_row.get('atcf_id', ''),
                            'Field':         field,
                            'HRDOBS_Value':  hrdobs_v,
                            'SHIPS_Value':   ships_v,
                            'Difference':    round(diff, 4),
                            'Units':         units,
                            'Tolerance':     tol,
                        })

                else:
                    # No matching SHIPS cycle — log advisory
                    qc_logs.append({
                        'Filename':       os.path.basename(input_h5),
                        'Group':          'ships_params',
                        'Variable':       'MATCH',
                        'Issue_Type':     (
                            f'Advisory: No SHIPS cycle matched for '
                            f'storm_id="{_storm_id}", datetime="{_storm_dt}" '
                            f'— ships_params group not written'
                        ),
                        'Bad_Obs_Valid':   0, 'Bad_Obs_Artifact': 0,
                        'Total_Obs_Valid': 0, 'Total_Obs_Artifact': 0,
                    })
                    # Structured no-match entry for dedicated CSV
                    ships_no_matches.append({
                        'Filename':    os.path.basename(input_h5),
                        'Storm_ID':    _storm_id,
                        'Datetime_UTC': _storm_dt,
                        'SHIPS_Key':   str(_ships_key) if _ships_key else 'unparseable',
                    })

            # ----------------------------------------------------------------
            # 5. Inventory record
            # ----------------------------------------------------------------
            if not scan_only:
                meta    = extract_filename_metadata(input_h5)

                # Dynamically extract requested SHIPS parameters for the inventory DB
                ships_inv_data = {var: np.nan for var in INVENTORY_SHIPS_VARS}
                if 'ships_params' in f_out:
                    sp_grp = f_out['ships_params']
                    for var in INVENTORY_SHIPS_VARS:
                        if var in sp_grp:
                            try:
                                val = sp_grp[var][0]
                                ships_inv_data[var] = float(val) if not np.isnan(val) else np.nan
                            except Exception:
                                pass

                inventory_entry = {
                    "Filename":              os.path.basename(output_h5),
                    "Virtual_Manifest":      np.nan,
                    "Storm":                 meta.get('storm_name', ''),
                    "Storm_ID":              meta.get('storm_id', ''),
                    "Storm_Datetime":        meta.get('storm_datetime', ''),
                    "Storm_Epoch":           meta.get('storm_epoch', ''),
                    "Lat":                   clat_val if clat_val is not None else np.nan,
                    "Lon":                   clon_val if clon_val is not None else np.nan,
                    "Intensity_ms":          intensity_val,
                    "MSLP_hPa":              mslp_val,
                    "TC_Category":           resolved_cat,
                    "Observation_Variables": ", ".join(sorted(all_vars)),
                    "Observation_Groups":    ", ".join(sorted(obs_groups)),
                    "Group_Counts_JSON":     json.dumps(group_counts),
                }
                
                # Merge the dynamically extracted SHIPS variables into the entry
                inventory_entry.update(ships_inv_data)

        null_ctx.close() if scan_only else None

        # ── Post-write check: delete empty files ──────────────────────
        if not obs_groups:
            if not scan_only and output_h5:
                try:
                    os.remove(output_h5)
                except OSError:
                    pass
            qc_logs.append({
                'Filename': os.path.basename(input_h5),
                'Group':    'ALL',
                'Variable': 'ALL',
                'Issue_Type': (
                    'SKIP: No observation or track groups survived QC '
                    '— empty AI-ready file deleted.'
                ),
                'Bad_Obs_Valid': 0, 'Bad_Obs_Artifact': 0,
                'Total_Obs_Valid': 0, 'Total_Obs_Artifact': 0,
            })
            inventory_entry = None

        # Abort if no observation groups survived: physically inspect the
        # output file rather than relying on in-memory tracking variables.
        if not scan_only and os.path.isfile(output_h5):
            valid_obs_survived = False
            try:
                with h5py.File(output_h5, 'r') as f_inspect:
                    actual_groups = list(f_inspect.keys())
                    
                # Define the contextual groups that don't count as "observations"
                context_groups = {'ships_params', 'storm_stats', 'track_best_track', 'track_spline_track'}
                
                # Check if anything besides context groups exists
                actual_obs_groups = [g for g in actual_groups if g not in context_groups]
                
                if len(actual_obs_groups) > 0:
                    valid_obs_survived = True
            except Exception:
                pass # If file is unreadable, it will naturally fail validation below
                
            if not valid_obs_survived:
                print(f"  ⏭️ Skipping {os.path.basename(output_h5)}: 0 observation arrays survived QC.")
                try:
                    os.remove(output_h5)
                except OSError:
                    pass
                return None, schema_updates, qc_logs, ships_mismatches, ships_no_matches

        # ── Post-write check: validate metadata completeness ──────────
        if not scan_only and output_h5 and inventory_entry is not None \
                and os.path.isfile(output_h5):
            meta_issues = []
            try:
                with h5py.File(output_h5, 'r') as f_check:
                    for field in sorted(CRITICAL_METADATA):
                        if field not in f_check.attrs:
                            meta_issues.append(
                                f"CRITICAL: '{field}' missing from metadata"
                            )
                        else:
                            # Safely cast to string to prevent float .strip() AttributeErrors
                            val = str(decode_attr(f_check.attrs[field])).strip()
                            if val in ('', 'NaN', 'nan'):
                                meta_issues.append(
                                    f"CRITICAL: '{field}' is '{val}'"
                                )

                    for field in sorted(EXPECTED_METADATA):
                        if field not in f_check.attrs:
                            meta_issues.append(
                                f"EXPECTED: '{field}' missing from metadata"
                            )
                        else:
                            # Safely cast to string to prevent float .strip() AttributeErrors
                            val = str(decode_attr(f_check.attrs[field])).strip()
                            if val == '':
                                meta_issues.append(
                                    f"EXPECTED: '{field}' is empty"
                                )

            except Exception:
                pass

            critical_failures = [m for m in meta_issues
                                 if m.startswith('CRITICAL')]

            if critical_failures:
                try:
                    os.remove(output_h5)
                except OSError:
                    pass
                for issue in meta_issues:
                    qc_logs.append({
                        'Filename': os.path.basename(input_h5),
                        'Group':    'GLOBAL_METADATA',
                        'Variable': 'ALL',
                        'Issue_Type': f'SKIP: {issue}',
                        'Bad_Obs_Valid': 0, 'Bad_Obs_Artifact': 0,
                        'Total_Obs_Valid': 0, 'Total_Obs_Artifact': 0,
                    })
                inventory_entry = None
            elif meta_issues:
                for issue in meta_issues:
                    qc_logs.append({
                        'Filename': os.path.basename(input_h5),
                        'Group':    'GLOBAL_METADATA',
                        'Variable': 'ALL',
                        'Issue_Type': f'Advisory: {issue}',
                        'Bad_Obs_Valid': 0, 'Bad_Obs_Artifact': 0,
                        'Total_Obs_Valid': 0, 'Total_Obs_Artifact': 0,
                    })

        if not scan_only and inventory_entry is not None and os.path.isfile(output_h5):
            manifest_file = generate_virtual_manifest(output_h5)
            if manifest_file:
                inventory_entry['Virtual_Manifest'] = os.path.basename(manifest_file)

    except Exception as e:
        print(f"  ❌ Failed {input_h5}: {e}")
        import traceback; traceback.print_exc()

    return inventory_entry, schema_updates, qc_logs, ships_mismatches, ships_no_matches


# =============================================================================
# MODE 3: REBUILD DB & SCHEMA FROM EXISTING AI-READY FILES
# =============================================================================

def extract_inventory_and_schema(h5_path):
    """Re-read a finished AI-ready file to rebuild inventory DB and schema."""
    schema_updates = {'global': set(), 'groups': {}}

    with h5py.File(h5_path, 'r') as f:
        for attr_name in f.attrs.keys():
            schema_updates['global'].add(attr_name)

        def get_attr(key, default="NaN"):
            val = f.attrs.get(key, default)
            return str(decode_attr(val)) if val != default else default

        meta      = extract_filename_metadata(h5_path)
        intensity = get_attr('storm_intensity_ms')
        mslp      = get_attr('storm_mslp_hpa')
        tc_cat    = get_attr('tc_category')

        center_str = get_attr('center_from_tc_vitals')
        clat, clon = np.nan, np.nan
        if center_str != "NaN" and "," in str(center_str):
            try:
                parts = center_str.split(",")
                clat = float(parts[0].strip())
                clon = float(parts[1].strip())
            except ValueError:
                pass

        obs_groups  = []
        group_counts = {}
        all_vars    = set()

        for g_name in f.keys():
            if not isinstance(f[g_name], h5py.Group):
                continue
            obs_groups.append(g_name)
            group_counts[g_name] = int(f[g_name].attrs.get('obs_count', 0))

            if g_name not in schema_updates['groups']:
                schema_updates['groups'][g_name] = {'attrs': set(), 'datasets': set()}
            for ak in f[g_name].attrs.keys():
                schema_updates['groups'][g_name]['attrs'].add(ak)
            for dset_name, dset in f[g_name].items():
                if isinstance(dset, h5py.Dataset):
                    schema_updates['groups'][g_name]['datasets'].add(dset_name)
                    all_vars.add(dset_name)

        # Dynamically extract requested SHIPS parameters for the inventory DB
        ships_inv_data = {var: np.nan for var in INVENTORY_SHIPS_VARS}
        if 'ships_params' in f:
            sp_grp = f['ships_params']
            for var in INVENTORY_SHIPS_VARS:
                if var in sp_grp:
                    try:
                        val = sp_grp[var][0]
                        ships_inv_data[var] = float(val) if not np.isnan(val) else np.nan
                    except Exception:
                        pass

        inventory_entry = {
            "Filename":              os.path.basename(h5_path),
            "Storm":                 meta.get('storm_name', ''),
            "Storm_ID":              meta.get('storm_id', ''),
            "Storm_Datetime":        meta.get('storm_datetime', ''),
            "Storm_Epoch":           meta.get('storm_epoch', ''),
            "Lat":                   clat,
            "Lon":                   clon,
            "Intensity_ms":          intensity,
            "MSLP_hPa":              mslp,
            "TC_Category":           tc_cat,
            "Observation_Variables": ", ".join(sorted(all_vars)),
            "Observation_Groups":    ", ".join(sorted(obs_groups)),
            "Group_Counts_JSON":     json.dumps(group_counts),
        }
        
        # Merge the dynamically extracted SHIPS variables into the entry
        inventory_entry.update(ships_inv_data)
        
    return inventory_entry, schema_updates


# =============================================================================
# SCHEMA SAVE
# =============================================================================

def save_schema(schema_master):
    rows = []
    for g in sorted(schema_master['global']):
        rows.append({"Item_Type": "Global Metadata Attribute", "Name": g})
    for grp, contents in sorted(schema_master['groups'].items()):
        rows.append({"Item_Type": "Data Group", "Name": grp})
        for a in sorted(contents['attrs']):
            rows.append({"Item_Type": f"Group Attribute ({grp})", "Name": a})
        for d in sorted(contents['datasets']):
            rows.append({"Item_Type": f"Dataset Variable ({grp})", "Name": d})

    pd.DataFrame(rows).to_csv(SCHEMA_REPORT, index=False)
    print("✅ Schema written to hrdobs_dataset_schema.csv")


# =============================================================================
# MODE 4: IDENTIFY DOUBLE ENTRIES
# =============================================================================

def identify_double_entries(input_dir):
    """
    Scan all original HDF5 files in input_dir and report cases where the same
    NHC storm designation (e.g. '02L') appears under more than one name in the
    same calendar year (e.g. 'TWO02L' vs 'BONNIE02L').
    """

    OUT_FILE = DOUBLE_ENTRIES_REPORT

    def emit(line="", file=None):
        print(line)
        file.write(line + "\n")

    input_files = glob.glob(os.path.join(input_dir, "**", "*.hdf5"), recursive=True)
    if not input_files:
        print(f"❌ No HDF5 files found in '{input_dir}'.")
        return

    print(f"  Scanning {len(input_files)} files in '{input_dir}'...\n")

    records = []
    skipped = 0

    for fp in input_files:
        meta = extract_filename_metadata(fp)
        if not meta:
            skipped += 1
            continue

        storm_id = meta.get('storm_id', '')
        dt_str   = meta.get('storm_datetime', '')

        nhc_match = re.search(r'(\d+[A-Z]+)$', storm_id)
        if not nhc_match:
            skipped += 1
            continue
        nhc_code = nhc_match.group(1)

        try:
            dt = datetime.datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:00Z")
        except ValueError:
            skipped += 1
            continue

        year  = dt.year
        cycle = dt.strftime("%m%d%H%M")

        records.append({
            'year':     year,
            'nhc_code': nhc_code,
            'storm_id': storm_id,
            'cycle':    cycle,
            'filepath': fp,
            'filename': os.path.basename(fp),
        })

    if skipped:
        print(f"  ⚠️  Skipped {skipped} file(s) with unparseable names.\n")

    if not records:
        print("  No parseable files found.")
        return

    df = pd.DataFrame(records)

    grouped = (
        df.groupby(['year', 'nhc_code'])['storm_id']
        .agg(lambda s: sorted(set(s)))
        .reset_index()
        .rename(columns={'storm_id': 'storm_ids'})
    )

    doubles = grouped[grouped['storm_ids'].apply(len) > 1].copy()
    doubles = doubles.sort_values(['year', 'nhc_code']).reset_index(drop=True)

    with open(OUT_FILE, 'w') as out_f:

        run_ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        emit(f"HRDOBS Double-Entry Report — generated {run_ts}", file=out_f)
        emit(f"Input directory : {os.path.abspath(input_dir)}", file=out_f)
        emit(f"Files scanned   : {len(input_files)}", file=out_f)
        emit("=" * 72, file=out_f)

        if doubles.empty:
            emit("✅ No double entries found — every NHC designation maps to a "
                 "single storm name within each year.", file=out_f)
            print(f"\n✅ Report written to {OUT_FILE}")
            return

        emit(f"⚠️  Found {len(doubles)} NHC designation(s) with multiple storm "
             f"names:\n", file=out_f)

        size_match_count   = 0
        size_differ_count  = 0

        current_year = None

        for _, row in doubles.iterrows():
            year      = row['year']
            nhc_code  = row['nhc_code']
            names     = row['storm_ids']

            cycles_by_name = {}
            for name in names:
                mask = (df['year'] == year) & (df['nhc_code'] == nhc_code) & \
                       (df['storm_id'] == name)
                cycles_by_name[name] = set(df.loc[mask, 'cycle'].tolist())

            all_cycles    = list(cycles_by_name.values())
            common_cycles = set.intersection(*all_cycles) if len(all_cycles) > 1 \
                            else set()

            if year != current_year:
                current_year = year
                emit(f"── {year} " + "─" * 54, file=out_f)

            group_mask = (df['year'] == year) & (df['nhc_code'] == nhc_code)
            group_files = sorted(df.loc[group_mask, 'filename'].tolist())
            for fname in group_files:
                emit(f"  {fname}", file=out_f)

            if common_cycles:
                path_lookup = {}
                for _, rec in df[group_mask].iterrows():
                    path_lookup[(rec['storm_id'], rec['cycle'])] = rec['filepath']

                size_verdicts = []
                for cycle in sorted(common_cycles):
                    sizes = {}
                    for name in names:
                        fpath = path_lookup.get((name, cycle))
                        if fpath:
                            try:
                                sizes[name] = os.path.getsize(fpath)
                            except OSError:
                                sizes[name] = None
                    unique_sizes = set(s for s in sizes.values() if s is not None)
                    if len(unique_sizes) == 1:
                        size_match_count += 1
                        size_verdicts.append("✅ identical size")
                    else:
                        size_differ_count += 1
                        size_verdicts.append("❌ sizes differ")

                size_summary = ', '.join(sorted(set(size_verdicts)))
                emit(f"  ⚠️  CYCLE OVERLAP ({len(common_cycles)} cycle(s)) — {size_summary}",
                     file=out_f)
            else:
                emit(f"  ✅ No cycle overlap — cycle times are disjoint.", file=out_f)

            emit("", file=out_f)

        summary = (f"Summary: {len(doubles)} double-entry group(s) found across "
                   f"{len(df['year'].unique())} year(s) / "
                   f"{len(input_files)} total file(s).")
        emit(f"\n{summary}", file=out_f)
        total_overlaps = size_match_count + size_differ_count
        if total_overlaps > 0:
            emit(f"  Overlapping cycles — same size : {size_match_count} / {total_overlaps}",
                 file=out_f)
            emit(f"  Overlapping cycles — diff size : {size_differ_count} / {total_overlaps}",
                 file=out_f)

    print(f"\n✅ Report written to {OUT_FILE}")


# =============================================================================
# MODE 5: TEMPORAL GAP CHECK
# =============================================================================

_NHC_SPELLED_NUMBERS = {
    'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE',
    'TEN', 'ELEVEN', 'TWELVE', 'THIRTEEN', 'FOURTEEN', 'FIFTEEN', 'SIXTEEN',
    'SEVENTEEN', 'EIGHTEEN', 'NINETEEN', 'TWENTY', 'TWENTYONE', 'TWENTYTWO',
    'TWENTYTHREE', 'TWENTYFOUR', 'TWENTYFIVE',
}

_NHC_INVEST = {'INVEST'}

_NHC_NUMBER_WORDS = _NHC_SPELLED_NUMBERS | _NHC_INVEST


def canonical_name(name_counts):
    """
    Given a dict of {storm_id: file_count} for one (year, nhc_code) group,
    return the canonical storm_id using a strict three-tier preference.
    """
    if len(name_counts) == 1:
        return next(iter(name_counts))

    def alpha_prefix(sid):
        m = re.match(r'([A-Za-z]+)', sid)
        return m.group(1).upper() if m else sid.upper()

    def best_in(pool):
        return max(pool, key=lambda sid: pool[sid])

    tier1 = {sid: cnt for sid, cnt in name_counts.items()
             if alpha_prefix(sid) not in _NHC_NUMBER_WORDS}
    if tier1:
        return best_in(tier1)

    tier2 = {sid: cnt for sid, cnt in name_counts.items()
             if alpha_prefix(sid) in _NHC_INVEST}
    if tier2:
        return best_in(tier2)

    return best_in(name_counts)


def check_temporal_gaps(input_dir):
    """
    For every storm in input_dir, build a unified deduplicated timeline of
    6-hourly cycle files and report any gaps between first and last cycle.
    """

    OUT_FILE = TEMPORAL_GAPS_REPORT
    GAP_HOURS = 6

    input_files = glob.glob(os.path.join(input_dir, "**", "*.hdf5"), recursive=True)
    if not input_files:
        print(f"❌ No HDF5 files found in '{input_dir}'.")
        return

    print(f"  Scanning {len(input_files)} files in '{input_dir}'...\n")

    records = []
    skipped = 0

    for fp in input_files:
        meta = extract_filename_metadata(fp)
        if not meta:
            skipped += 1
            continue

        storm_id = meta.get('storm_id', '')
        dt_str   = meta.get('storm_datetime', '')

        nhc_match = re.search(r'(\d+[A-Z]+)$', storm_id)
        if not nhc_match:
            skipped += 1
            continue
        nhc_code = nhc_match.group(1)

        try:
            dt = datetime.datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:00Z")
        except ValueError:
            skipped += 1
            continue

        records.append({
            'year':     dt.year,
            'nhc_code': nhc_code,
            'storm_id': storm_id,
            'dt':       dt,
        })

    if skipped:
        print(f"  ⚠️  Skipped {skipped} file(s) with unparseable names.\n")

    if not records:
        print("  No parseable files found.")
        return

    df = pd.DataFrame(records)

    name_counts_df = (
        df.groupby(['year', 'nhc_code', 'storm_id'])
        .size()
        .reset_index(name='count')
    )

    canonical = {}
    for (year, nhc_code), grp in name_counts_df.groupby(['year', 'nhc_code']):
        nc_map = dict(zip(grp['storm_id'], grp['count']))
        canonical[(year, nhc_code)] = canonical_name(nc_map)

    df['canonical'] = df.apply(
        lambda r: canonical[(r['year'], r['nhc_code'])], axis=1
    )

    unified = (
        df.groupby(['year', 'nhc_code', 'canonical', 'dt'])
        .size()
        .reset_index(name='file_count')
        .sort_values(['year', 'nhc_code', 'dt'])
        .reset_index(drop=True)
    )

    with open(OUT_FILE, 'w') as out_f:

        def emit(line=""):
            print(line)
            out_f.write(line + "\n")

        def femit(line=""):
            out_f.write(line + "\n")

        run_ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        emit(f"HRDOBS Temporal Gap Report — generated {run_ts}")
        emit(f"Input directory : {os.path.abspath(input_dir)}")
        emit(f"Files scanned   : {len(input_files)}")
        emit(f"Expected spacing: {GAP_HOURS}h")
        emit("=" * 72)

        total_storms  = 0
        storms_clean  = 0
        storms_gaps   = 0
        total_gaps    = 0

        for (year, nhc_code), grp in unified.groupby(['year', 'nhc_code']):
            canon  = grp['canonical'].iloc[0]
            times  = sorted(grp['dt'].tolist())
            n_files = len(times)

            first_t = times[0]
            last_t  = times[-1]

            expected = []
            t = first_t
            while t <= last_t:
                expected.append(t)
                t += datetime.timedelta(hours=GAP_HOURS)

            present_set   = set(times)
            missing_times = [t for t in expected if t not in present_set]

            total_storms += 1
            fmt = "%b %d %Hz"

            print("")
            print(f"  Year  : {year}")
            print(f"  Storm : {canon}  (NHC: {nhc_code})")
            print(f"  Files : {n_files}  |  "
                  f"First: {first_t.strftime(fmt)}  |  "
                  f"Last : {last_t.strftime(fmt)}")

            if not missing_times:
                print(f"  ✅ No gaps — all {len(expected)} expected cycles present.")
                storms_clean += 1
            else:
                storms_gaps  += 1
                total_gaps   += len(missing_times)
                missing_str   = ', '.join(t.strftime(fmt) for t in missing_times)
                print(f"  ⚠️  {len(missing_times)} gap(s) out of "
                      f"{len(expected)} expected cycles:")
                print(f"    Missing: {missing_str}")

            print("-" * 72)

            femit("")
            femit(f"  Year  : {year}")
            femit(f"  Storm : {canon}  (NHC: {nhc_code})")
            femit(f"  Files : {n_files}  |  "
                  f"First: {first_t.strftime(fmt)}  |  "
                  f"Last : {last_t.strftime(fmt)}")

            if not missing_times:
                femit(f"  ✅ No gaps — all {len(expected)} expected cycles present.")
            else:
                femit(f"  ⚠️  {len(missing_times)} gap(s) out of "
                      f"{len(expected)} expected cycles:")

            for exp_t in expected:
                label = exp_t.strftime(fmt)
                if exp_t in present_set:
                    femit(f"    {label}")
                else:
                    femit(f"    {label}  -> missing")

            femit("-" * 72)

        emit("")
        emit(f"Summary: {total_storms} storm(s) checked  |  "
             f"{storms_clean} complete  |  "
             f"{storms_gaps} with gap(s)  |  "
             f"{total_gaps} total missing cycle(s)")

    print(f"\n✅ Report written to {OUT_FILE}")


# =============================================================================
# MODE 5: RENAME DOUBLE-ENTRY FILES
# =============================================================================

def rename_double_entries(input_dir):
    """
    For every overlapping (year, nhc_code) group, list all files and show
    any renames required to align them to the canonical storm name.
    No files are modified until the user confirms.
    """

    OUT_FILE = RENAME_PLAN_REPORT

    input_files = glob.glob(os.path.join(input_dir, "**", "*.hdf5"), recursive=True)
    if not input_files:
        print(f"❌ No HDF5 files found in '{input_dir}'.")
        return

    print(f"  Scanning {len(input_files)} files in '{input_dir}'...\n")

    records = []
    skipped = 0

    for fp in input_files:
        meta = extract_filename_metadata(fp)
        if not meta:
            skipped += 1
            continue

        storm_id = meta.get('storm_id', '')
        dt_str   = meta.get('storm_datetime', '')

        nhc_match = re.search(r'(\d+[A-Z]+)$', storm_id)
        if not nhc_match:
            skipped += 1
            continue
        nhc_code = nhc_match.group(1)

        try:
            dt = datetime.datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:00Z")
        except ValueError:
            skipped += 1
            continue

        records.append({
            'year':     dt.year,
            'nhc_code': nhc_code,
            'storm_id': storm_id,
            'filepath': fp,
            'filename': os.path.basename(fp),
        })

    if skipped:
        print(f"  ⚠️  Skipped {skipped} file(s) with unparseable names.\n")

    if not records:
        print("  No parseable files found.")
        return

    df = pd.DataFrame(records)

    name_counts_df = (
        df.groupby(['year', 'nhc_code', 'storm_id'])
        .size()
        .reset_index(name='count')
    )

    multi = (
        name_counts_df.groupby(['year', 'nhc_code'])
        .filter(lambda g: g['storm_id'].nunique() > 1)
    )

    if multi.empty:
        print("✅ No double entries found — nothing to rename.")
        return

    canon_map = {}
    for (year, nhc_code), grp in multi.groupby(['year', 'nhc_code']):
        nc_map = dict(zip(grp['storm_id'], grp['count']))
        canon_map[(year, nhc_code)] = canonical_name(nc_map)

    overlap_keys = set(canon_map.keys())
    df = df[df.apply(
        lambda r: (r['year'], r['nhc_code']) in overlap_keys, axis=1
    )].copy()
    df['canonical'] = df.apply(
        lambda r: canon_map[(r['year'], r['nhc_code'])], axis=1
    )

    def make_new_filename(old_fname, old_storm_id, canon_id):
        return old_fname.replace(
            f'HRDOBS_{old_storm_id}.', f'HRDOBS_{canon_id}.', 1
        )

    df['needs_rename'] = df['storm_id'] != df['canonical']
    df['new_filename'] = df.apply(
        lambda r: make_new_filename(r['filename'], r['storm_id'], r['canonical'])
                  if r['needs_rename'] else r['filename'],
        axis=1
    )

    with open(OUT_FILE, 'w') as out_f:

        def emit(line=""):
            print(line)
            out_f.write(line + "\n")

        run_ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        emit(f"HRDOBS Rename Plan — generated {run_ts}")
        emit(f"Input directory : {os.path.abspath(input_dir)}")
        emit(f"Files scanned   : {len(input_files)}")
        emit(f"(PREVIEW — files will be renamed only after confirmation)")
        emit("=" * 72)

        n_to_rename = int(df['needs_rename'].sum())
        emit(f"Overlapping groups : {len(canon_map)}")
        emit(f"Files to rename    : {n_to_rename}")
        emit("")

        df_sorted = df.sort_values(['year', 'filename']).reset_index(drop=True)

        current_year = None
        for _, rec in df_sorted.iterrows():
            if rec['year'] != current_year:
                current_year = rec['year']
                emit(f"── {current_year} " + "─" * 54)

            if rec['needs_rename']:
                emit(f"  {rec['filename']}  ->  {rec['new_filename']}")
            else:
                emit(f"  {rec['filename']}")

        emit("")
        emit(f"Summary: {n_to_rename} file(s) would be renamed across "
             f"{len(canon_map)} overlapping group(s).")

    print(f"\n✅ Rename plan written to {OUT_FILE}")

    if n_to_rename == 0:
        return

    print(f"\n⚠️  About to rename {n_to_rename} file(s) in '{input_dir}'.")
    confirm = input("Type YES to proceed, or anything else to abort: ").strip()
    if confirm != 'YES':
        print("Aborted — no files were renamed.")
        return

    renamed_ok  = 0
    renamed_err = 0
    for _, rec in df[df['needs_rename']].iterrows():
        old_path = rec['filepath']
        new_path = os.path.join(os.path.dirname(old_path), rec['new_filename'])
        try:
            os.rename(old_path, new_path)
            renamed_ok += 1
        except OSError as e:
            print(f"  ❌ Failed to rename {rec['filename']}: {e}")
            renamed_err += 1

    print(f"\n✅ Renamed {renamed_ok} file(s) successfully."
          + (f"  {renamed_err} error(s)." if renamed_err else ""))


# =============================================================================
# MODE 7: IDENTIFY SPLINE TRACK INCONSISTENCIES
# =============================================================================

def check_spline_track_altitudes(input_dir):
    """
    For every original HDF5 file, inspect the spline track group and attempt
    to determine the flight-level altitude from the corresponding raw .fix file.
    Produces a diagnostic CSV (spline_track_altitude_report.csv).
    """

    OUT_FILE = SPLINE_ALT_REPORT

    input_files = sorted(
        glob.glob(os.path.join(input_dir, "**", "*.hdf5"), recursive=True)
    )
    if not input_files:
        print(f"❌ No HDF5 files found in '{input_dir}'.")
        return

    filtered_files = [f for f in input_files if should_process_file(f)]
    n_skipped_basin = len(input_files) - len(filtered_files)
    print(f"Found {len(input_files)} original HDF5 files.")
    if n_skipped_basin > 0:
        print(f"  Skipped {n_skipped_basin} file(s) outside "
              f"{_BASIN_LABEL.get(BASIN_FILTER, BASIN_FILTER)} basin.")
    print(f"  Processing {len(filtered_files)} file(s).\n")

    fix_cache = {}

    report_rows = []
    counters = {
        'total':              0,
        'no_spline':          0,
        'spline_empty':       0,
        'fix_file_missing':   0,
        'no_fixes_in_window': 0,
        'altitude_ok':        0,
        'mixed_altitudes':    0,
    }

    for i, input_path in enumerate(filtered_files):
        meta     = extract_filename_metadata(input_path)
        filename = os.path.basename(input_path)
        storm_id = meta.get('storm_id', '')
        year_str = meta.get('storm_datetime', '')[:4]
        year     = int(year_str) if year_str else 0
        basin    = detect_basin(storm_id)

        counters['total'] += 1

        row = {
            'Filename':           filename,
            'Storm_ID':           storm_id,
            'Basin':              basin,
            'Cycle_Datetime':     meta.get('storm_datetime', ''),
            'TC_Category':        '',
            'Intensity_ms':       '',
            'Spline_Present':     False,
            'Spline_Points':      0,
            'Spline_Start':       '',
            'Spline_End':         '',
            'Window_Start':       '',
            'Window_End':         '',
            'Fix_File':           '',
            'Fix_File_Found':     False,
            'Fixes_In_Window':    0,
            'Altitude_Values':    '',
            'Mean_Altitude_mb':   '',
            'FL_Groups':          '',
            'FL_N_Obs':           0,
            'FL_Mean_P_mb':       '',
            'FL_Per_Aircraft':    '',
            'Status':             '',
        }

        try:
            with h5py.File(input_path, 'r') as f_in:

                raw_cat_val = ''
                intensity_val = ''
                if 'storm stats' in f_in:
                    ss = f_in['storm stats']
                    if 'tc_category' in ss.attrs:
                        raw_cat_val = decode_attr(ss.attrs['tc_category'])
                    if 'storm_intensity_ms' in ss.attrs:
                        intensity_val = decode_attr(ss.attrs['storm_intensity_ms'])
                    elif 'intensity' in ss.attrs:
                        intensity_val = decode_attr(ss.attrs['intensity'])

                if (not intensity_val or intensity_val in ('NaN', '')) and \
                   'track' in f_in and 'best track' in f_in['track']:
                    bt = f_in['track']['best track']
                    match = re.search(r'[\._](\d{4})(\d{2})(\d{2})(\d{2})', input_path)
                    if match and 'time' in bt and 'vmax' in bt:
                        target_str = f"{match.group(1)}{match.group(2)}{match.group(3)}{match.group(4)}0000"
                        try:
                            target_dt = pd.to_datetime(target_str, format="%Y%m%d%H%M%S")
                            bt_times = bt['time'][:]
                            bt_dts = pd.to_datetime(bt_times, format='%Y%m%d%H%M%S',
                                                    errors='coerce')
                            valid = bt_dts.notna()
                            if valid.any():
                                diffs = np.abs((bt_dts[valid] - target_dt).total_seconds())
                                best_idx = int(np.argmin(diffs.to_numpy()))
                                if diffs.to_numpy()[best_idx] <= 10800:
                                    intensity_val = str(float(bt['vmax'][best_idx]))
                        except Exception:
                            pass

                resolved_cat, _ = resolve_tc_category(
                    raw_cat_val if raw_cat_val else None,
                    intensity_val if intensity_val else 'NaN'
                )
                row['TC_Category'] = resolved_cat
                row['Intensity_ms'] = intensity_val if intensity_val else 'NaN'

                spline_grp = None
                if 'track' in f_in and 'spline track' in f_in['track']:
                    spline_grp = f_in['track']['spline track']
                elif 'track' in f_in and 'spline_track' in f_in['track']:
                    spline_grp = f_in['track']['spline_track']

                if spline_grp is None:
                    row['Status'] = 'NO_SPLINE_GROUP'
                    counters['no_spline'] += 1
                    report_rows.append(row)
                    continue

                if 'time' not in spline_grp:
                    row['Status'] = 'NO_SPLINE_TIME'
                    counters['no_spline'] += 1
                    report_rows.append(row)
                    continue

                time_arr = spline_grp['time'][:]
                if np.issubdtype(time_arr.dtype, np.number):
                    grp_missing = spline_grp.attrs.get('missing_value', None)
                    time_arr = replace_missing_values(time_arr, grp_missing)

                time_arr = time_arr[~np.isnan(time_arr)]

                if len(time_arr) == 0:
                    row['Status'] = 'SPLINE_EMPTY'
                    counters['spline_empty'] += 1
                    report_rows.append(row)
                    continue

                row['Spline_Present'] = True
                row['Spline_Points']  = len(time_arr)

                fl_info = extract_flight_level_pressure(f_in)
                if fl_info['groups']:
                    row['FL_Groups']        = ', '.join(fl_info['groups'])
                    row['FL_N_Obs']         = fl_info['overall_n_obs']
                    row['FL_Mean_P_mb']     = fl_info['overall_mean_p_mb']
                    row['FL_Per_Aircraft']   = json.dumps(fl_info['per_aircraft'])

                try:
                    t_first = pd.to_datetime(str(int(time_arr[0])),
                                             format='%Y%m%d%H%M%S')
                    t_last  = pd.to_datetime(str(int(time_arr[-1])),
                                             format='%Y%m%d%H%M%S')
                    row['Spline_Start'] = t_first.strftime('%Y-%m-%d %H:%M')
                    row['Spline_End']   = t_last.strftime('%Y-%m-%d %H:%M')
                except Exception:
                    row['Spline_Start'] = 'PARSE_ERROR'
                    row['Spline_End']   = 'PARSE_ERROR'

        except Exception as e:
            row['Status'] = f'HDF5_READ_ERROR: {e}'
            report_rows.append(row)
            continue

        cycle_dt_str = meta.get('storm_datetime', '')
        try:
            cycle_dt = datetime.datetime.strptime(cycle_dt_str,
                                                  "%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, TypeError):
            row['Status'] = 'CYCLE_TIME_PARSE_ERROR'
            report_rows.append(row)
            continue

        window_start = cycle_dt - datetime.timedelta(hours=3)
        window_end   = cycle_dt + datetime.timedelta(hours=3)
        row['Window_Start'] = window_start.strftime('%Y-%m-%d %H:%M')
        row['Window_End']   = window_end.strftime('%Y-%m-%d %H:%M')

        fix_path = find_fix_file(storm_id, year)
        row['Fix_File'] = os.path.basename(fix_path) if fix_path else ''
        row['Fix_File_Found'] = fix_path is not None

        if fix_path is None:
            row['Status'] = 'FIX_FILE_MISSING'
            counters['fix_file_missing'] += 1
            report_rows.append(row)
            continue

        cache_key = fix_path
        if cache_key not in fix_cache:
            try:
                fix_cache[cache_key] = parse_fix_file(fix_path)
            except Exception as e:
                row['Status'] = f'FIX_PARSE_ERROR: {e}'
                report_rows.append(row)
                continue
        all_fixes = fix_cache[cache_key]

        mean_alt, n_fixes, alt_counts = compute_spline_altitude(
            all_fixes, window_start, window_end
        )

        row['Fixes_In_Window'] = n_fixes

        if n_fixes == 0:
            row['Status'] = 'NO_FIXES_IN_WINDOW'
            counters['no_fixes_in_window'] += 1
            report_rows.append(row)
            continue

        row['Altitude_Values']  = json.dumps(alt_counts)
        row['Mean_Altitude_mb'] = round(mean_alt, 1)

        if len(alt_counts) == 1:
            row['Status'] = 'OK'
            counters['altitude_ok'] += 1
        else:
            row['Status'] = 'MIXED_ALTITUDES'
            counters['mixed_altitudes'] += 1

        report_rows.append(row)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(filtered_files)} files...")

    df_report = pd.DataFrame(report_rows)

    resolved_mask = df_report['Status'].isin(['OK', 'MIXED_ALTITUDES'])
    resolved_df   = df_report[resolved_mask].copy()
    resolved_df['_alt'] = pd.to_numeric(resolved_df['Mean_Altitude_mb'],
                                         errors='coerce')
    cat_alt_lookup = {}
    for cat, grp in resolved_df.groupby('TC_Category'):
        a = grp['_alt'].dropna()
        if len(a) > 0:
            cat_alt_lookup[cat] = round(float(a.median()), 0)

    storm_resolved = {}
    for _, r in resolved_df.iterrows():
        sid = r['Storm_ID']
        alt = r['_alt']
        dt  = r['Cycle_Datetime']
        if pd.notna(alt) and dt:
            storm_resolved.setdefault(sid, []).append((dt, alt))

    for sid in storm_resolved:
        parsed = []
        for dt_str, alt in storm_resolved[sid]:
            try:
                dt = datetime.datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ")
                parsed.append((dt, alt))
            except (ValueError, TypeError):
                pass
        storm_resolved[sid] = parsed

    rec_alts    = []
    rec_sources = []

    for _, row in df_report.iterrows():
        status = row['Status']

        if status in ('OK', 'MIXED_ALTITUDES'):
            alt = pd.to_numeric(row['Mean_Altitude_mb'], errors='coerce')
            rec_alts.append(round(float(alt), 1) if pd.notna(alt) else '')
            rec_sources.append('fix_file')
            continue

        if not row.get('Spline_Present', False):
            rec_alts.append('')
            rec_sources.append('')
            continue

        sid      = row['Storm_ID']
        dt_str   = row['Cycle_Datetime']
        cat      = row['TC_Category']

        if sid in storm_resolved and storm_resolved[sid]:
            try:
                gap_dt = datetime.datetime.strptime(dt_str,
                                                    "%Y-%m-%dT%H:%M:%SZ")
                nearest_alt = min(
                    storm_resolved[sid],
                    key=lambda x: abs((x[0] - gap_dt).total_seconds())
                )[1]
                rec_alts.append(round(nearest_alt, 1))
                rec_sources.append('nearest_neighbor')
                continue
            except (ValueError, TypeError):
                pass

        if cat in cat_alt_lookup:
            rec_alts.append(cat_alt_lookup[cat])
            rec_sources.append('category_default')
            continue

        fl_p = pd.to_numeric(row.get('FL_Mean_P_mb', ''), errors='coerce')
        if pd.notna(fl_p) and fl_p > 0:
            rec_alts.append(round(fl_p, 1))
            rec_sources.append('fl_pressure')
            continue

        rec_alts.append('')
        rec_sources.append('none')

    df_report['Recommended_Alt_mb'] = rec_alts
    df_report['Recommended_Source'] = rec_sources

    df_report.to_csv(OUT_FILE, index=False)

    print()
    print("=" * 68)
    print("Spline Track Altitude Diagnostic Summary")
    print("=" * 68)
    print(f"  Basin filter          : {_BASIN_LABEL.get(BASIN_FILTER, BASIN_FILTER)}")
    print(f"  Files processed       : {counters['total']}")
    print(f"  No spline track       : {counters['no_spline']}")
    print(f"  Spline empty          : {counters['spline_empty']}")
    print(f"  Fix file missing      : {counters['fix_file_missing']}")
    print(f"  No fixes in window    : {counters['no_fixes_in_window']}")
    print(f"  Altitude OK (uniform) : {counters['altitude_ok']}")
    print(f"  Mixed altitudes       : {counters['mixed_altitudes']}")
    print("-" * 68)

    has_spline = df_report[df_report['Spline_Present'] == True]
    has_fl     = has_spline[has_spline['FL_N_Obs'] > 0]
    no_fl      = has_spline[has_spline['FL_N_Obs'] == 0]
    print(f"  Files with spline track           : {len(has_spline)}")
    print(f"  … with flight-level pressure data : {len(has_fl)}")
    print(f"  … WITHOUT flight-level data       : {len(no_fl)}")

    nf_with_fl = df_report[
        (df_report['Status'] == 'NO_FIXES_IN_WINDOW') &
        (df_report['FL_N_Obs'] > 0)
    ]
    nf_without_fl = df_report[
        (df_report['Status'] == 'NO_FIXES_IN_WINDOW') &
        (df_report['FL_N_Obs'] == 0)
    ]
    print(f"\n  NO_FIXES_IN_WINDOW with FL fallback  : {len(nf_with_fl)}")
    print(f"  NO_FIXES_IN_WINDOW without FL data   : {len(nf_without_fl)}")
    print("-" * 68)

    has_spline_t = df_report[df_report['Spline_Present'] == True]
    rec_counts = has_spline_t['Recommended_Source'].value_counts()
    print(f"  Recommended altitude sources:")
    for src, cnt in rec_counts.items():
        if src:
            print(f"    {src:<25s} : {cnt:5d}")
    no_rec = has_spline_t[
        (has_spline_t['Recommended_Alt_mb'] == '') |
        (has_spline_t['Recommended_Alt_mb'].isna())
    ]
    if len(no_rec) > 0:
        print(f"    {'NO RECOMMENDATION':<25s} : {len(no_rec):5d}")

    print("=" * 68)
    print(f"\n✅ Full report → {OUT_FILE}")

    DIAG_FILE = SPLINE_GAP_DIAGNOSTICS

    resolved = df_report[df_report['Status'].isin(['OK', 'MIXED_ALTITUDES'])].copy()
    resolved['alt'] = pd.to_numeric(resolved['Mean_Altitude_mb'], errors='coerce')

    cat_order = ['TD', 'TS', 'H1', 'H2', 'H3', 'H4', 'H5',
                 'SS', 'SD', 'EX', 'LO', 'DB', 'WV', 'HU', 'NaN']
    cats_present = [c for c in cat_order if c in resolved['TC_Category'].values]
    for c in sorted(resolved['TC_Category'].unique()):
        if c not in cats_present:
            cats_present.append(c)

    with open(DIAG_FILE, 'w') as diag_f:

        def emit(line=""):
            diag_f.write(line + "\n")

        run_ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        emit(f"Spline Track Gap Diagnostics — generated {run_ts}")
        emit(f"Basin filter : {_BASIN_LABEL.get(BASIN_FILTER, BASIN_FILTER)}")
        emit(f"Input        : {input_dir}")
        emit(f"Fix files    : {FIXES_DIR}")
        emit()

        emit(f"{'='*68}")
        emit(f"ALTITUDE BY TC CATEGORY (resolved files, n={len(resolved)})")
        emit(f"{'='*68}")
        emit(f"  {'Category':<8s} {'Files':>6s} {'Mean':>7s} {'Median':>7s} "
             f"{'Min':>7s} {'Max':>7s}  Common Levels")
        emit(f"  {'-'*8} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7}  {'-'*25}")

        cat_alt_lookup = {}
        for cat in cats_present:
            sub = resolved[resolved['TC_Category'] == cat]
            if len(sub) == 0:
                continue
            a = sub['alt'].dropna()
            if len(a) == 0:
                continue
            top_levels = a.round(0).value_counts().head(3)
            levels_str = ', '.join(f"{int(lv)}" for lv in top_levels.index)
            cat_alt_lookup[cat] = float(a.median())
            emit(f"  {cat:<8s} {len(a):6d} {a.mean():7.0f} {a.median():7.0f} "
                 f"{a.min():7.0f} {a.max():7.0f}  {levels_str}")

        gap_files = df_report[df_report['Status'] == 'NO_FIXES_IN_WINDOW'].copy()
        if not gap_files.empty:
            gap_files['fl_p'] = pd.to_numeric(gap_files['FL_Mean_P_mb'], errors='coerce')

            emit(f"\n{'='*68}")
            emit(f"GAP FILE DIAGNOSTICS ({len(gap_files)} files)")
            emit(f"{'='*68}")

            for sid in sorted(gap_files['Storm_ID'].unique()):
                storm_gaps = gap_files[gap_files['Storm_ID'] == sid].sort_values('Cycle_Datetime')
                storm_resolved_sub = resolved[resolved['Storm_ID'] == sid]

                if len(storm_resolved_sub) > 0:
                    storm_alts = storm_resolved_sub['alt'].dropna()
                    nn_info = (f"storm avg={storm_alts.mean():.0f} mb, "
                               f"range={storm_alts.min():.0f}–{storm_alts.max():.0f}")
                else:
                    nn_info = "no resolved cycles"

                n_with_fl = int((storm_gaps['FL_N_Obs'] > 0).sum())

                emit(f"\n  {sid}  ({len(storm_gaps)} gaps, "
                     f"{len(storm_resolved_sub)} resolved, {n_with_fl} with FL)")
                emit(f"    Neighbor data: {nn_info}")

                for _, grow in storm_gaps.iterrows():
                    dt_short = grow['Cycle_Datetime'][5:16] if grow['Cycle_Datetime'] else '?'
                    cat = grow['TC_Category']
                    fl_str = f"FL={grow['fl_p']:.0f}" if pd.notna(grow['fl_p']) else "no FL"
                    cat_default = cat_alt_lookup.get(cat, '')
                    cat_str = f"cat_def={cat_default:.0f}" if cat_default else "no cat_def"
                    rec_alt = grow.get('Recommended_Alt_mb', '')
                    rec_src = grow.get('Recommended_Source', '')
                    emit(f"      {dt_short}  cat={cat:<4s}  {fl_str:<12s}  "
                         f"{cat_str:<16s}  → {rec_alt} mb ({rec_src})")

        missing_fix = df_report[df_report['Status'] == 'FIX_FILE_MISSING']
        if not missing_fix.empty:
            unique_storms = missing_fix['Storm_ID'].unique()
            emit(f"\n{'='*68}")
            emit(f"MISSING FIX FILES ({len(unique_storms)} storms, "
                 f"{len(missing_fix)} cycles)")
            emit(f"{'='*68}")
            for sid in sorted(unique_storms):
                storm_rows = missing_fix[missing_fix['Storm_ID'] == sid]
                n_cycles = len(storm_rows)
                emit(f"  {sid:25s}  ({n_cycles} cycles)")

        has_spline_d = df_report[df_report['Spline_Present'] == True]

        emit(f"\n{'='*68}")
        emit(f"RECOMMENDATION SUMMARY")
        emit(f"{'='*68}")

        rec_source_counts = has_spline_d['Recommended_Source'].value_counts()
        for src, cnt in rec_source_counts.items():
            if src:
                emit(f"  {src:<25s} : {cnt:5d} files")

        no_rec = has_spline_d[
            (has_spline_d['Recommended_Alt_mb'] == '') |
            (has_spline_d['Recommended_Alt_mb'].isna())
        ]
        if len(no_rec) > 0:
            emit(f"  {'NO RECOMMENDATION':<25s} : {len(no_rec):5d} files")
            for _, r in no_rec.iterrows():
                emit(f"    {r['Filename']}")

        rec_alts_all = pd.to_numeric(has_spline_d['Recommended_Alt_mb'],
                                     errors='coerce').dropna()
        if len(rec_alts_all) > 0:
            emit(f"\n  Recommended altitude distribution "
                 f"(all {len(rec_alts_all)} files):")
            alt_dist = rec_alts_all.round(0).value_counts().sort_index()
            for alt, cnt in alt_dist.items():
                bar = '█' * (cnt // 10) + ('▌' if cnt % 10 >= 5 else '')
                emit(f"    {alt:6.0f} mb : {cnt:4d}  {bar}")

    print(f"✅ Gap diagnostics → {DIAG_FILE}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"\n📌 Basin filter: {_BASIN_LABEL.get(BASIN_FILTER, BASIN_FILTER)}")
    print(f"   (Change BASIN_FILTER at top of script to 'ATL', 'EPAC', or 'ALL')\n")

    print("Select Run Mode:")
    print("  1 — Identify Double Entries (same NHC code, multiple storm names)")
    print("  2 — Rename Double-Entry Files (preview plan, then confirm to execute)")
    print("  3 — Check Temporal Gaps (missing 6-hourly cycles per storm)")
    print("  4 — Spline Track Altitude Diagnostic (flight-level assignment)")
    print("  5 — QC Scan Only (no files written)")
    print("  6 — Full Processing (QC + Convert to AI-ready + Build DB/Schema)")
    print("  7 — Build DB and Schema Only (from existing AI-ready files)")
    mode = input("Enter 1–7: ").strip()

    if mode not in ('1', '2', '3', '4', '5', '6', '7'):
        print("Invalid selection.")
        return

    if mode == '1':
        print(f"\n🔍 Mode 1: Scanning '{INPUT_DIR}' for double-entry storm names...\n")
        identify_double_entries(INPUT_DIR)
        return

    if mode == '2':
        print(f"\n🔍 Mode 2: Building rename plan for '{INPUT_DIR}'...\n")
        rename_double_entries(INPUT_DIR)
        return

    if mode == '3':
        print(f"\n🔍 Mode 3: Checking temporal gaps in '{INPUT_DIR}'...\n")
        check_temporal_gaps(INPUT_DIR)
        return

    if mode == '4':
        print(f"\n🔍 Mode 4: Spline Track Altitude Diagnostic "
              f"({_BASIN_LABEL.get(BASIN_FILTER, BASIN_FILTER)})")
        print(f"   Input  : {INPUT_DIR}")
        print(f"   Fixes  : {FIXES_DIR}\n")
        check_spline_track_altitudes(INPUT_DIR)
        return

    if mode == '5':
        print("\n🔍 Mode 5: QC Scan Only — no files will be written (Simulation Mode).")
        input_files = glob.glob(os.path.join(INPUT_DIR, "**", "*.hdf5"), recursive=True)
        input_files = [f for f in input_files if should_process_file(f)]
        print(f"Processing {len(input_files)} {_BASIN_LABEL.get(BASIN_FILTER, BASIN_FILTER)} HDF5 files.")

        ships_lookup = load_ships_lookup(SHIPS_CSV_PATH)

        all_qc_logs = []
        all_ships_mismatches = []
        all_ships_no_matches  = []
        error_sim_logs = []
        skipped_files = []
        simulated_written = 0
        skip_reason_counts = {}
        
        for i, input_path in enumerate(input_files):
            filename = os.path.basename(input_path)
            _, _, qc, mm, nm = convert_universal(input_path, None, scan_only=True,
                                                  ships_lookup=ships_lookup,
                                                  error_sim_logs=error_sim_logs)
            all_ships_mismatches.extend(mm)
            all_ships_no_matches.extend(nm)
            
            is_skipped = False
            if qc:
                all_qc_logs.extend(qc)
                skip_reasons = [
                    log['Issue_Type'] for log in qc
                    if log.get('Issue_Type', '').startswith('SKIP:')
                    or log.get('Issue_Type', '').startswith('WARNING: No observation')
                ]
                if skip_reasons:
                    skipped_files.append((filename, skip_reasons))
                    is_skipped = True
                    for reason in skip_reasons:
                        skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + 1
            
            if not is_skipped:
                simulated_written += 1
                
            if (i + 1) % 100 == 0:
                print(f"  Scanned {i+1}/{len(input_files)} files...")
                
        if all_qc_logs:
            pd.DataFrame(all_qc_logs).to_csv(QC_FORENSICS_REPORT, index=False)
            print("✅ QC Forensics Report → qc_forensics_report.csv")
        else:
            print("✅ No QC issues found.")

        if error_sim_logs:
            pd.DataFrame(error_sim_logs).to_csv(ERROR_SIM_REPORT, index=False)
            print(f"✅ Error Assignment Simulation → error_assignment_simulation.csv ({len(error_sim_logs)} actions logged)")

        _mismatch_csv = SHIPS_MISMATCH_REPORT
        if all_ships_mismatches:
            pd.DataFrame(all_ships_mismatches).to_csv(_mismatch_csv, index=False)
            print(f"⚠️  SHIPS metadata mismatches → {_mismatch_csv} "
                  f"({len(all_ships_mismatches)} entr(ies) across "
                  f"{len(set(r['Filename'] for r in all_ships_mismatches))} file(s))")
        else:
            print(f"✅ No SHIPS/HRDOBS metadata mismatches — {_mismatch_csv} not written.")

        _nomatch_csv = SHIPS_NOMATCH_REPORT
        if all_ships_no_matches:
            pd.DataFrame(all_ships_no_matches).to_csv(_nomatch_csv, index=False)
            print(f"⚠️  SHIPS no-match cycles → {_nomatch_csv} "
                  f"({len(all_ships_no_matches)} cycle(s) across "
                  f"{len(set(r['Storm_ID'] for r in all_ships_no_matches))} storm(s))")
        else:
            print(f"✅ All HRDOBS cycles matched a SHIPS entry — {_nomatch_csv} not written.")

        if skipped_files:
            print(f"\n{'='*68}")
            print(f"⚠️  {len(skipped_files)} FILE(S) WOULD BE SKIPPED")
            print(f"   ({simulated_written} files would be written successfully)")
            print(f"{'='*68}")
            for fname, reasons in skipped_files:
                print(f"\n  ❌ {fname}")
                for reason in reasons:
                    print(f"     {reason}")
            print(f"\n{'='*68}")
            
            print("\n📊 Summary of Skip Reasons:")
            for reason, count in sorted(skip_reason_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {count} time(s): {reason}")
            print(f"{'='*68}")
            
        print(f"\n  Input files scanned : {len(input_files)}")
        print(f"  Simulated written   : {simulated_written}")
        print(f"  Simulated skipped   : {len(skipped_files)}")
        print("=" * 68)
        return

    if mode == '6':
        if os.path.exists(OUTPUT_DIR):
            print(f"\n⚠️  Mode 6 will delete and recreate '{OUTPUT_DIR}' before processing.")
            confirm = input("Type YES to continue, or anything else to abort: ").strip()
            if confirm != "YES":
                print("Aborted.")
                return
        print(f"\n🚀 Mode 6: Full Conversion — clearing {OUTPUT_DIR}...")
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "HRDOBS_hdf5"), exist_ok=True)

        spline_alt_lookup = {}
        if os.path.isfile(SPLINE_ALT_REPORT):
            try:
                alt_df = pd.read_csv(SPLINE_ALT_REPORT)
                for _, r in alt_df.iterrows():
                    fname = r.get('Filename', '')
                    rec   = r.get('Recommended_Alt_mb', '')
                    if fname and rec != '' and pd.notna(rec):
                        try:
                            spline_alt_lookup[fname] = float(rec)
                        except (ValueError, TypeError):
                            pass
                print(f"📌 Loaded spline track altitudes for "
                      f"{len(spline_alt_lookup)} files from {SPLINE_ALT_REPORT}")
            except Exception as e:
                print(f"⚠️  Could not load {SPLINE_ALT_REPORT}: {e}")
                print("   Spline track pressure will NOT be written.")
        else:
            print(f"\n⚠️  {SPLINE_ALT_REPORT} not found.")
            print("   Mode 6 can compute spline track altitudes on the fly")
            print("   (same as Mode 4), but this skips the manual review step.")
            compute = input("   Compute altitudes now? (YES to proceed, "
                            "anything else to skip): ").strip()
            if compute == 'YES':
                print(f"\n   Running spline track altitude analysis...")
                check_spline_track_altitudes(INPUT_DIR)
                if os.path.isfile(SPLINE_ALT_REPORT):
                    try:
                        alt_df = pd.read_csv(SPLINE_ALT_REPORT)
                        for _, r in alt_df.iterrows():
                            fname = r.get('Filename', '')
                            rec   = r.get('Recommended_Alt_mb', '')
                            if fname and rec != '' and pd.notna(rec):
                                try:
                                    spline_alt_lookup[fname] = float(rec)
                                except (ValueError, TypeError):
                                    pass
                        print(f"\n📌 Computed and loaded spline track altitudes "
                              f"for {len(spline_alt_lookup)} files.")
                    except Exception as e:
                        print(f"⚠️  Could not load generated report: {e}")
                        print("   Spline track pressure will NOT be written.")
                else:
                    print("⚠️  Altitude report was not generated.")
                    print("   Spline track pressure will NOT be written.")
            else:
                print("   Aborted — run Mode 4 first to review altitude "
                      "assignments, then re-run Mode 6.")
                return

        input_files = glob.glob(os.path.join(INPUT_DIR, "**", "*.hdf5"), recursive=True)
        input_files = [f for f in input_files if should_process_file(f)]
        print(f"Processing {len(input_files)} {_BASIN_LABEL.get(BASIN_FILTER, BASIN_FILTER)} HDF5 files.")

        ships_lookup = load_ships_lookup(SHIPS_CSV_PATH)
        ships_matched = 0

        inventory_list = []
        schema_master  = {'global': set(), 'groups': {}}
        all_qc_logs        = []
        all_ships_mismatches = []
        all_ships_no_matches  = []
        alt_injected       = 0
        skipped_files  = []

        for i, input_path in enumerate(input_files):
            rel_path = os.path.relpath(input_path, INPUT_DIR)
            rel_dir  = os.path.dirname(rel_path)
            out_dir  = os.path.join(OUTPUT_DIR, "HRDOBS_hdf5", rel_dir)
            os.makedirs(out_dir, exist_ok=True)

            filename = os.path.basename(input_path)
            out_name = re.sub(r'\.(hdf5|h5)$', r'_AI_READY.\1', filename)
            out_path = os.path.join(out_dir, out_name)

            file_alt = spline_alt_lookup.get(filename, None)
            if file_alt is not None:
                alt_injected += 1

            inv, sch, qc, mm, nm = convert_universal(input_path, out_path,
                                                 scan_only=False,
                                                 spline_alt_mb=file_alt,
                                                 ships_lookup=ships_lookup)
            all_ships_mismatches.extend(mm)
            all_ships_no_matches.extend(nm)

            if qc:
                ships_matched += sum(
                    1 for log in qc
                    if log.get('Group') == 'ships_params'
                    and 'No SHIPS cycle matched' not in log.get('Issue_Type', '')
                    and log.get('Issue_Type', '').startswith('Advisory:')
                    is False
                )
            if sch and 'ships_params' in sch.get('groups', {}):
                ships_matched += 1

            if qc:
                all_qc_logs.extend(qc)
                skip_reasons = [
                    log['Issue_Type'] for log in qc
                    if log.get('Issue_Type', '').startswith('SKIP:')
                    or log.get('Issue_Type', '').startswith('WARNING: No observation')
                ]
                if skip_reasons:
                    skipped_files.append((filename, skip_reasons))

            if inv:
                inventory_list.append(inv)
                schema_master['global'].update(sch['global'])
                for g_name, g_data in sch['groups'].items():
                    if g_name not in schema_master['groups']:
                        schema_master['groups'][g_name] = {'attrs': set(), 'datasets': set()}
                    schema_master['groups'][g_name]['attrs'].update(g_data['attrs'])
                    schema_master['groups'][g_name]['datasets'].update(g_data['datasets'])

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(input_files)} files...")

        if all_qc_logs:
            pd.DataFrame(all_qc_logs).to_csv(QC_FORENSICS_REPORT, index=False)
            print("✅ QC Forensics Report → qc_forensics_report.csv")
        else:
            print("✅ No QC issues found.")

        print(f"✅ Spline track pressure injected in {alt_injected} file(s).")

        if ships_lookup:
            ships_written = sum(
                1 for log in all_qc_logs
                if log.get('Group') == 'ships_params'
                and log.get('Variable') == 'MATCH'   
            )
            n_no_match = ships_written   
            n_matched  = len(inventory_list) - n_no_match
            n_total    = len(input_files)
            print(f"✅ SHIPS params written: {n_matched}/{n_total} files matched")
            if n_no_match > 0:
                print(f"   ⚠️  {n_no_match} file(s) had no matching SHIPS cycle "
                      f"(see qc_forensics_report.csv for details)")

            _mismatch_csv = SHIPS_MISMATCH_REPORT
            if all_ships_mismatches:
                pd.DataFrame(all_ships_mismatches).to_csv(_mismatch_csv, index=False)
                n_files = len(set(r['Filename'] for r in all_ships_mismatches))
                print(f"\n{'='*68}")
                print(f"⚠️  SHIPS METADATA MISMATCHES — "
                      f"{len(all_ships_mismatches)} instance(s) across {n_files} file(s)")
                print(f"   (advisory only — no data was modified)")
                print(f"   Details → {_mismatch_csv}")
                print(f"{'='*68}")
                for r in all_ships_mismatches:
                    print(f"  {r['Filename']}")
                    print(f"    {r['Field']}: HRDOBS={r['HRDOBS_Value']} "
                          f"SHIPS={r['SHIPS_Value']} "
                          f"diff={r['Difference']} {r['Units']} "
                          f"(tol={r['Tolerance']})")
                print(f"{'='*68}")
            else:
                print(f"✅ No SHIPS/HRDOBS metadata mismatches — {_mismatch_csv} not written.")

            _nomatch_csv = SHIPS_NOMATCH_REPORT
            if all_ships_no_matches:
                pd.DataFrame(all_ships_no_matches).to_csv(_nomatch_csv, index=False)
                n_nm_storms = len(set(r['Storm_ID'] for r in all_ships_no_matches))
                print(f"\n⚠️  SHIPS no-match cycles → {_nomatch_csv} "
                      f"({len(all_ships_no_matches)} cycle(s) across "
                      f"{n_nm_storms} storm(s))")
            else:
                print(f"✅ All HRDOBS cycles matched a SHIPS entry "
                      f"— {_nomatch_csv} not written.")

        pd.DataFrame(inventory_list).to_csv(INVENTORY_DB, index=False)
        print(f"✅ Inventory DB → hrdobs_inventory_db.csv ({len(inventory_list)} files).")
        save_schema(schema_master)

        if skipped_files:
            n_skipped = len(skipped_files)
            n_written = len(inventory_list)
            print(f"\n{'='*68}")
            print(f"⚠️  {n_skipped} FILE(S) SKIPPED — not written to output")
            print(f"   ({n_written} files written successfully)")
            print(f"{'='*68}")
            for fname, reasons in skipped_files:
                print(f"\n  ❌ {fname}")
                for reason in reasons:
                    print(f"     {reason}")
            print(f"\n{'='*68}")
            print()
            print("=" * 68)
            print(f"⚠️  {len(skipped_files)} FILE(S) SKIPPED — not included "
                  f"in AI-ready dataset:")
            print("=" * 68)
            for fname, reasons in skipped_files:
                print(f"\n  {fname}")
                for reason in reasons:
                    print(f"    → {reason}")
            print()
            print(f"  Input files scanned : {len(input_files)}")
            print(f"  Files written       : {len(inventory_list)}")
            print(f"  Files skipped       : {len(skipped_files)}")
            print("=" * 68)
        else:
            print(f"\n✅ All {len(input_files)} files processed successfully.")

        return

    if mode == '7':
        print(f"\n🔍 Mode 7: Rebuilding DB and Schema from {OUTPUT_DIR}...")
        target_dir = os.path.join(OUTPUT_DIR, "HRDOBS_hdf5")
        ai_files   = glob.glob(os.path.join(target_dir, "**", "*.hdf5"), recursive=True)

        if not ai_files:
            print(f"❌ No files found in {target_dir}. Run Mode 6 first.")
            return

        ai_files = [f for f in ai_files if should_process_file(f)]
        print(f"Processing {len(ai_files)} {_BASIN_LABEL.get(BASIN_FILTER, BASIN_FILTER)} AI-ready files.")

        inventory_list = []
        schema_master  = {'global': set(), 'groups': {}}

        for i, fp in enumerate(ai_files):
            inv, sch = extract_inventory_and_schema(fp)
            inventory_list.append(inv)
            schema_master['global'].update(sch['global'])
            for g_name, g_data in sch['groups'].items():
                if g_name not in schema_master['groups']:
                    schema_master['groups'][g_name] = {'attrs': set(), 'datasets': set()}
                schema_master['groups'][g_name]['attrs'].update(g_data['attrs'])
                schema_master['groups'][g_name]['datasets'].update(g_data['datasets'])
            if (i + 1) % 500 == 0:
                print(f"   Scanned {i+1}/{len(ai_files)} files...")

        pd.DataFrame(inventory_list).to_csv(INVENTORY_DB, index=False)
        print(f"✅ Inventory DB written ({len(inventory_list)} entries).")
        save_schema(schema_master)
        return


if __name__ == "__main__":
    main()