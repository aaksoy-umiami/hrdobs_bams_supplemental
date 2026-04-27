# -*- coding: utf-8 -*-
"""
ships_to_csv.py
===============
Parses a SHIPS lsdiag predictor file (any basin, any year range) and writes
one CSV row per storm cycle containing storm metadata from the HEAD record
and 25 environmental predictor values at t=0 (current cycle time only).

USAGE
-----
    python ships_to_csv.py <input_lsdiag_file> [output_csv_file] [start_year] [end_year]

    start_year and end_year are optional integers.  When provided, only cycles
    whose year falls within [start_year, end_year] inclusive are written.
    The parser stops reading the file as soon as it encounters the first HEAD
    line whose year exceeds end_year, so processing is efficient even on
    full multi-decade lsdiag files.

    If output_csv_file is omitted, the output is written to the same directory
    as the input file with a .csv extension.

FILE FORMAT
-----------
Each data row contains 31 values in fixed-width 5-character fields covering
times -12, -6, 0, 6, 12, ... 168 hr relative to the cycle time.  The t=0
value is always at character positions [10:15].  The row label is the
rightmost alphabetically-starting token on the line.

Special-format rows (PSLV, MTPW, HIST, IR**, PC**, LAST) are skipped because
they do not follow the standard 31-column layout.

Missing values (9999 or blank) are written as empty strings in the CSV.

SCALING
-------
Stored integer values are converted to physical units as follows:

Variable     Stored unit              Conversion              Output unit
CSST         deg C * 10              / 10                    deg C
CD20/CD26    m (no scale)            --                       m
COHC         kJ/cm2 (no scale)       --                       kJ/cm2
DTL          km (no scale)           --                       km
OAGE/NAGE    hr * 10                 / 10                    hr
INCV         kt (no scale)           --                       kt
SHRD/SHDC    kt * 10                 / 10                    kt
SHTD/SDDC    deg (no scale)          --                       deg
RHLO/MD/HI   % (no scale)            --                       %
VMPI         kt (no scale)           --                       kt
PENV/PENC    (hPa - 1000) * 10       / 10 + 1000             hPa
Z850         s^-1 * 10^7             -- (kept as stored)      10^-7 s^-1
D200         s^-1 * 10^7             -- (kept as stored)      10^-7 s^-1
U200         kt * 10                 / 10                    kt
DSST/NSST    deg C * 10              / 10                    deg C
NOHC         kJ/cm2 (no scale)       --                       kJ/cm2

NOTES ON SPECIFIC VARIABLES
----------------------------
INCV at t=0 : intensity change from -6 to 0 hr -- the 6-hour intensification
              rate immediately before the current cycle.  Missing when the
              storm has no prior 6-hour record.

Z850/D200   : retained in the stored integer units (x10^7) rather than
              converted to SI values, which are awkward in CSV format.
              Divide by 10,000,000 to obtain s^-1.

PENV/PENC   : the stored value is (hPa - 1000) x 10, so very negative values
              (e.g. -690) represent pressures below 1000 hPa by 69 hPa.

Longitude   : the HEAD line stores longitude as degrees West.  It is written
              to the CSV as a negative value (degrees East) to match the
              HRDOBS sign convention.

MATCHING WITH HRDOBS
--------------------
The CSV columns 'hrdobs_storm_id' and 'datetime_utc' together form the
unique key that matches each row to an HRDOBS AI-Ready cycle file:

    HRDOBS filename : HRDOBS_<hrdobs_storm_id>.<YYYYMMDDHHMMSS>_AI_READY.hdf5
    HRDOBS metadata : storm_id       = hrdobs_storm_id
                      storm_datetime = datetime_utc

DATASET REFERENCE
-----------------
For a full description of the HRDOBS AI-Ready dataset structure, variable
definitions, and quality-control procedures, refer to the accompanying
manuscript and dataset documentation.
"""

import sys
import os
import csv
import re
from pathlib import Path


# =============================================================================
# CONFIGURATION -- target variables and their scaling
# =============================================================================

# Each entry: SHIPS_label -> (csv_column_name, scale_fn, description)
# scale_fn receives the raw integer and returns the physical value.

MISSING = 9999

TARGET_VARS = {
    # -- Storm state and history ----------------------------------------------
    'TYPE':  ('type',
              lambda v: v,
              'Storm type: 0=wave/remnant/dissipating, 1=tropical, '
              '2=subtropical, 3=extratropical'),

    'INCV':  ('incv_kt',
              lambda v: v,
              'Intensity change -6 to 0 hr (kt); past 6-hour intensification rate'),

    'CSST':  ('csst_degc',
              lambda v: round(v / 10.0, 1),
              'Climatological SST along track (deg C)'),

    'CD20':  ('cd20_m',
              lambda v: v,
              'Climatological depth of 20 deg C isotherm (m)'),

    'CD26':  ('cd26_m',
              lambda v: v,
              'Climatological depth of 26 deg C isotherm (m)'),

    'COHC':  ('cohc_kjcm2',
              lambda v: v,
              'Climatological ocean heat content relative to 26 deg C (kJ/cm2)'),

    'DTL':   ('dtl_km',
              lambda v: v,
              'Distance to nearest major land mass (km)'),

    'OAGE':  ('oage_hr',
              lambda v: round(v / 10.0, 1),
              'Ocean age: time storm occupied area within 100 km (hr)'),

    'NAGE':  ('nage_hr',
              lambda v: round(v / 10.0, 1),
              'Intensity-weighted ocean age (hr); equals OAGE when Vmax=100 kt'),

    # -- Atmospheric environment ----------------------------------------------
    'SHRD':  ('shrd_kt',
              lambda v: round(v / 10.0, 1),
              '850-200 hPa shear magnitude (kt), r=200-800 km'),

    'SHTD':  ('shtd_deg',
              lambda v: v,
              'Heading of 850-200 hPa shear vector (deg); 90=westerly'),

    'SHDC':  ('shdc_kt',
              lambda v: round(v / 10.0, 1),
              '850-200 hPa vortex-removed shear magnitude (kt), r=0-500 km '
              'relative to 850 hPa vortex center'),

    'SDDC':  ('sddc_deg',
              lambda v: v,
              'Heading of vortex-removed shear vector (deg)'),

    'RHLO':  ('rhlo_pct',
              lambda v: v,
              '850-700 hPa relative humidity (%), r=200-800 km'),

    'RHMD':  ('rhmd_pct',
              lambda v: v,
              '700-500 hPa relative humidity (%), r=200-800 km'),

    'RHHI':  ('rhhi_pct',
              lambda v: v,
              '500-300 hPa relative humidity (%), r=200-800 km'),

    'VMPI':  ('vmpi_kt',
              lambda v: v,
              'Maximum potential intensity from Kerry Emanuel equation (kt)'),

    'PENV':  ('penv_hpa',
              lambda v: round(v / 10.0 + 1000.0, 1),
              'Average environmental surface pressure (hPa), r=200-800 km'),

    'PENC':  ('penc_hpa',
              lambda v: round(v / 10.0 + 1000.0, 1),
              'Azimuthally averaged surface pressure at outer vortex edge (hPa)'),

    'Z850':  ('z850_1e7_per_s',
              lambda v: v,
              '850 hPa vorticity (10^-7 s^-1), r=0-1000 km; divide by 1e7 for SI'),

    'D200':  ('d200_1e7_per_s',
              lambda v: v,
              '200 hPa divergence (10^-7 s^-1); divide by 1e7 for SI'),

    'U200':  ('u200_kt',
              lambda v: round(v / 10.0, 1),
              '200 hPa zonal wind (kt), r=200-800 km; negative=easterly'),

    # -- Ocean analysis -------------------------------------------------------
    'DSST':  ('dsst_degc',
              lambda v: round(v / 10.0, 1),
              'Daily Reynolds SST along track (deg C)'),

    'NSST':  ('nsst_degc',
              lambda v: round(v / 10.0, 1),
              'NCODA analysis SST (deg C)'),

    'NOHC':  ('nohc_kjcm2',
              lambda v: v,
              'NCODA ocean heat content relative to 26 deg C isotherm (kJ/cm2)'),
}

# Lines that do not follow the standard 31-column layout -- skip entirely
SKIP_LABELS = {
    'TIME', 'HIST', 'PSLV', 'MTPW',
    'IRXX', 'IR00', 'IRM1', 'IRM3',
    'PC00', 'PCM1', 'PCM3',
    'LAST',
}

# CSV column names whose physical values must always be >= 0.
# Any negative value after scaling is treated as a missing-value sentinel and
# written as an empty string.  This guards against non-standard sentinel values
# (e.g., -999 in DTL) that appear in some source files.
#
# Variables intentionally excluded from this set (negative values are
# physically meaningful):
#   incv_kt        -- weakening produces negative intensity change
#   z850_1e7_per_s -- anticyclonic vorticity is negative
#   d200_1e7_per_s -- convergence is negative divergence
#   u200_kt        -- easterly upper-level flow is negative
#   csst/dsst/nsst_degc -- SST can be near 0 in high-latitude cases
NON_NEGATIVE_COLS = {
    # Ocean thermal structure
    'cd20_m', 'cd26_m', 'cohc_kjcm2', 'nohc_kjcm2',
    # Storm geometry and ocean age
    'dtl_km', 'oage_hr', 'nage_hr',
    # Wind shear magnitudes (always >= 0 by definition)
    'shrd_kt', 'shdc_kt',
    # Headings (0-360 deg, always >= 0)
    'shtd_deg', 'sddc_deg',
    # Relative humidity (0-100 %, always >= 0)
    'rhlo_pct', 'rhmd_pct', 'rhhi_pct',
    # MPI (always >= 0, set to 0 when SST below threshold)
    'vmpi_kt',
}

# Basin letter -> HRDOBS filename suffix mapping
_BASIN_SUFFIX = {'AL': 'L', 'EP': 'E', 'CP': 'C', 'WP': 'W', 'IO': 'I', 'SH': 'S'}


# =============================================================================
# HELPERS
# =============================================================================

def extract_label(line):
    """
    Return the variable label from a SHIPS data line.

    The label is the rightmost whitespace-delimited token that starts with
    a letter.  This correctly handles age-indicator suffixes like 'RSST    3'
    (returns 'RSST') and standard lines like 'VMAX' (returns 'VMAX').
    """
    for token in reversed(line.split()):
        if token and token[0].isalpha():
            return token.upper()
    return None


def extract_t0(line):
    """
    Return the raw integer t=0 value from a standard SHIPS data line,
    or None if the field is blank or equals 9999 (missing).

    The t=0 column is always at fixed character positions [10:15] in
    the 5-char-per-field fixed-width format.
    """
    if len(line) < 15:
        return None
    field = line[10:15].strip()
    if not field:
        return None
    try:
        v = int(field)
        return None if v == MISSING else v
    except ValueError:
        return None


def parse_head(line):
    """
    Parse a SHIPS HEAD line and return a dict of storm metadata.

    HEAD format (space-delimited):
        <NAME4> <YYMMDD> <HH> <VMAX_kt> <LAT_degN> <LON_degW> <MSLP_hPa> <ATCF_ID> HEAD

    Returns None if the line cannot be parsed.
    """
    parts = line.split()
    # Minimum expected: name, date, hour, vmax, lat, lon, mslp, atcf_id, 'HEAD'
    if len(parts) < 9:
        return None

    try:
        name      = parts[0].strip()
        yymmdd    = parts[1].strip()
        hour_str  = parts[2].strip().zfill(2)
        vmax_kt   = float(parts[3])
        lat_degN  = float(parts[4])
        lon_degW  = float(parts[5])
        mslp_hpa  = float(parts[6])
        atcf_id   = parts[7].strip()

        # Parse date -- YYMMDD where YY is 2-digit (assume 1900s<2000, 00s+>=2000)
        yy = int(yymmdd[0:2])
        mm = int(yymmdd[2:4])
        dd = int(yymmdd[4:6])
        hh = int(hour_str)
        year = 2000 + yy if yy < 50 else 1900 + yy

        datetime_utc = f"{year:04d}-{mm:02d}-{dd:02d}T{hh:02d}:00:00Z"

        # Derive HRDOBS-style storm_id from ATCF ID
        # ATCF format: <BB><NN><YYYY> e.g. AL012014
        # HRDOBS format: <NAME><NN><suffix> e.g. ARTH01L
        basin_code  = atcf_id[:2].upper()          # e.g. 'AL'
        storm_num   = atcf_id[2:4]                 # e.g. '01'
        basin_sfx   = _BASIN_SUFFIX.get(basin_code, '?')
        hrdobs_id   = f"{name}{storm_num}{basin_sfx}"  # e.g. 'ARTH01L'

        return {
            'storm_name':      name,
            'atcf_id':         atcf_id,
            'hrdobs_storm_id': hrdobs_id,
            'year':            year,
            'month':           mm,
            'day':             dd,
            'hour':            hh,
            'datetime_utc':    datetime_utc,
            'vmax_hd_kt':      vmax_kt,
            'lat_hd_degN':     lat_degN,
            'lon_hd_degE':     -lon_degW,   # convert W->E for HRDOBS convention
            'mslp_hd_hpa':     mslp_hpa,
        }
    except (ValueError, IndexError):
        return None


# =============================================================================
# MAIN PARSER
# =============================================================================

def parse_ships_file(input_path, start_year=None, end_year=None):
    """
    Parse an entire SHIPS lsdiag file and return a list of cycle dicts,
    one per HEAD...LAST block.

    Parameters
    ----------
    start_year : int or None
        If provided, only cycles from the first cycle of this year onwards
        are included.  Cycles in earlier years are skipped entirely.
    end_year : int or None
        If provided, only cycles up to the last cycle of this year are
        included.  The parser stops reading as soon as it encounters a
        HEAD line whose year exceeds end_year.
    """
    cycles = []
    current_meta   = None
    current_vars   = {}
    in_window      = True   # set False once we pass end_year

    with open(input_path, 'r') as fh:
        for raw_line in fh:
            line = raw_line.rstrip()
            if not line.strip():
                continue

            label = extract_label(line)
            if label is None:
                continue

            # -- Cycle boundaries ----------------------------------------
            if label == 'HEAD':
                meta = parse_head(line)
                if meta is None:
                    current_meta = None
                    current_vars = {}
                    continue

                cycle_year = meta['year']

                # Stop reading once we've passed end_year
                if end_year is not None and cycle_year > end_year:
                    in_window = False
                    current_meta = None
                    current_vars = {}
                    break

                # Skip cycles before start_year
                if start_year is not None and cycle_year < start_year:
                    current_meta = None
                    current_vars = {}
                    continue

                current_meta = meta
                current_vars = {}
                continue

            if label == 'LAST':
                if current_meta is not None and in_window:
                    row = dict(current_meta)
                    for ships_label, (col_name, scale_fn, _) in TARGET_VARS.items():
                        raw = current_vars.get(ships_label)
                        if raw is not None:
                            val = scale_fn(raw)
                            # Non-negative check: treat negative physical values
                            # as missing-value sentinels for constrained variables.
                            if col_name in NON_NEGATIVE_COLS and                                     isinstance(val, (int, float)) and val < 0:
                                row[col_name] = ''
                            else:
                                row[col_name] = val
                        else:
                            row[col_name] = ''
                    cycles.append(row)
                current_meta = None
                current_vars = {}
                continue

            # -- Skip non-standard lines ----------------------------------
            if label in SKIP_LABELS:
                continue
            if current_meta is None:
                continue

            # -- Extract t=0 for target variables ------------------------
            if label in TARGET_VARS:
                t0 = extract_t0(line)
                if t0 is not None:
                    current_vars[label] = t0

    return cycles


# =============================================================================
# CSV OUTPUT
# =============================================================================

def write_csv(cycles, output_path):
    """Write the list of cycle dicts to a CSV file."""
    if not cycles:
        print("No cycles parsed -- output file not written.")
        return

    # Column order: metadata first, then variables in TARGET_VARS order
    meta_cols = [
        'storm_name', 'atcf_id', 'hrdobs_storm_id',
        'year', 'month', 'day', 'hour', 'datetime_utc',
        'vmax_hd_kt', 'lat_hd_degN', 'lon_hd_degE', 'mslp_hd_hpa',
    ]
    var_cols = [col for col, _, _ in TARGET_VARS.values()]
    all_cols = meta_cols + var_cols

    with open(output_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=all_cols, extrasaction='ignore')

        # Header row
        writer.writeheader()

        # Write a human-readable units/description block as commented lines
        # at the top of the file so tools that skip '#' lines load cleanly.
        fh.write('#\n')
        fh.write('# Column descriptions and units:\n')
        fh.write('#   storm_name      : 4-character NHC storm name\n')
        fh.write('#   atcf_id         : ATCF storm ID (e.g. AL012014)\n')
        fh.write('#   hrdobs_storm_id : HRDOBS-format storm ID (e.g. ARTH01L) '
                 '-- use with datetime_utc to match HRDOBS cycles\n')
        fh.write('#   year/month/day/hour : cycle date and UTC hour\n')
        fh.write('#   datetime_utc    : ISO-8601 cycle time (matches HRDOBS storm_datetime)\n')
        fh.write('#   vmax_hd_kt      : max surface wind from HEAD (kt)\n')
        fh.write('#   lat_hd_degN     : storm latitude from HEAD (deg N)\n')
        fh.write('#   lon_hd_degE     : storm longitude from HEAD (deg E, negative=west)\n')
        fh.write('#   mslp_hd_hpa     : min sea-level pressure from HEAD (hPa)\n')
        for ships_label, (col_name, _, description) in TARGET_VARS.items():
            fh.write(f'#   {col_name:<22s}: [{ships_label}] {description}\n')
        fh.write('#\n')

        # Data rows
        writer.writerows(cycles)

    print(f"[OK] Written {len(cycles)} cycles -> {output_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python ships_to_csv.py <input_lsdiag_file> [output_csv_file] "
              "[start_year] [end_year]")
        print("  start_year / end_year are optional integers, e.g. 2010 2023")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.isfile(input_path):
        print(f"[ERROR] File not found: {input_path}")
        sys.exit(1)

    output_path = sys.argv[2] if len(sys.argv) >= 3 else \
                  str(Path(input_path).with_suffix('.csv'))

    start_year = end_year = None
    try:
        if len(sys.argv) >= 4:
            start_year = int(sys.argv[3])
        if len(sys.argv) >= 5:
            end_year   = int(sys.argv[4])
    except ValueError:
        print("[ERROR] start_year and end_year must be integers.")
        sys.exit(1)

    if start_year and end_year and start_year > end_year:
        print(f"[ERROR] start_year ({start_year}) must be <= end_year ({end_year}).")
        sys.exit(1)

    year_msg = ""
    if start_year and end_year:
        year_msg = f" (years {start_year}-{end_year})"
    elif start_year:
        year_msg = f" (from {start_year})"
    elif end_year:
        year_msg = f" (through {end_year})"

    print(f"Parsing: {input_path}{year_msg}")
    cycles = parse_ships_file(input_path, start_year=start_year, end_year=end_year)
    print(f"  Cycles found: {len(cycles)}")

    write_csv(cycles, output_path)


if __name__ == '__main__':
    main()
