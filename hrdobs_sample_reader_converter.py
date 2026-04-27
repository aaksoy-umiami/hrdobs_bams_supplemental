#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hrdobs_sample_reader_converter.py
==================================
Supplementary reader and converter for the HRDOBS AI-Ready HDF5 dataset (v1.0).

OVERVIEW
--------
This script provides an interactive command-line interface for inspecting and
exporting the contents of HRDOBS AI-Ready HDF5 files.  It is intended as a
lightweight companion tool that requires only standard scientific Python
libraries (h5py, numpy) and no knowledge of the HDF5 file format.

USAGE
-----
1. Place this script in the same directory as your HRDOBS .hdf5 files,
   or run it from that directory.

2. Execute the script:
       python hrdobs_sample_reader_converter.py

3. When prompted, select which files to inspect using one of the following
   formats (file numbers are shown at startup):
       0          — process all files found in the directory
       1          — a single file
       1, 3, 5    — a comma-separated list of files
       2-6        — a hyphenated range of files
       1, 3-5, 8  — any combination of the above

4. Select an output format:
       1  Expanded listing — prints full metadata, SHIPS parameters, and an
          observation-group inventory for each selected file individually.
       2  Compact listing  — same content as option 1 but in a condensed
          tabular layout, one file at a time.
       3  Tabled overview  — aggregates metadata, SHIPS parameters, and
          group statistics across all selected files into side-by-side
          comparison tables.
       4  Single CSV export — writes the complete contents of all selected
          files (metadata, SHIPS parameters, and all data arrays) into one
          CSV file named hrdobs_file_contents.csv.
       5  Individual CSV export — same as option 4 but produces a separate
          CSV file for each selected file, named hrdobs_file_contents_N.csv.

CSV FORMAT (options 4 and 5)
-----------------------------
Each file's block within the CSV is structured as follows:
  - One row with the source filename.
  - A metadata section: two-column rows of (parameter_name, value).
  - A SHIPS section: three-column rows of (short_name, long_name, value).
  - One block per observation group: a schema row (group name, column count,
    row count), a header row of variable names, and one data row per
    observation.  Time values are written as ISO-8601 strings.
  - Separator rows ("---") between sections and "===" at the end of each file.

DEPENDENCIES
------------
    h5py >= 2.10
    numpy >= 1.18

DATASET REFERENCE
-----------------
For a full description of the HRDOBS AI-Ready dataset structure, variable
definitions, and quality-control procedures, refer to the accompanying
manuscript and dataset documentation.
"""

import h5py
import numpy as np
import glob
import re
import csv
import os
from datetime import datetime, timezone, timedelta

# =============================================================================
# CONFIGURATION
# =============================================================================

# Coordinate and geometry variables defined in the HRDOBS dataset schema.
# These are excluded from the "data variable" count when summarizing groups.
KNOWN_COORDS = {
    'time', 'lat', 'lon', 'latitude', 'longitude',
    'clat', 'clon',           # track center position
    'height', 'altitude',     # vertical coordinates
    'elev',                   # TDR elevation angle
    'az',                     # TDR azimuth angle
    'p', 'pres', 'pressure',  # pressure levels
    'ght',                    # geopotential height
    'rmw',                    # radius of maximum wind
    'rr',                     # SFMR rain rate
}

# Complete list of observation groups defined in the HRDOBS dataset schema.
# Files may contain any subset of these groups depending on which instruments
# were active during the reconnaissance mission.
EXPECTED_GROUPS = [
    "dropsonde_ghawk",
    "dropsonde_noaa42", "dropsonde_noaa43", "dropsonde_noaa49", "dropsonde_usaf",
    "flight_level_hdobs_noaa42", "flight_level_hdobs_noaa43",
    "flight_level_hdobs_noaa49", "flight_level_hdobs_usaf",
    "sfmr_noaa42", "sfmr_noaa43", "sfmr_usaf",
    "ships_params",
    "tdr_noaa42", "tdr_noaa43", "tdr_noaa49",
    "track_best_track", "track_spline_track", "track_vortex_message",
]

# Ordered sequence of metadata fields shown in all display and export modes.
META_ORDER = [
    'version_number',
    'storm_id',
    'storm_datetime_year', 'storm_datetime_month',
    'storm_datetime_day',  'storm_datetime_hour',
    'storm_intensity_ms', 'storm_mslp_hpa', 'tc_category',
    'center_from_tc_vitals_lat', 'center_from_tc_vitals_lon',
    'radius_of_maximum_wind_km',
    'storm_motion_speed_kt', 'storm_motion_heading_deg',
    'time_coverage_start', 'time_coverage_end',
    'existing_groups', 'expected_groups',
]

# Reference epoch for CF-convention time decoding.
_CF_EPOCH = datetime(1900, 1, 1, tzinfo=timezone.utc)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def decode_attr(val):
    """
    Decode an HDF5 attribute value to a plain Python type.

    Converts bytes and numpy byte strings to UTF-8 text, numpy string or
    object arrays to lists of decoded strings, and numpy numeric scalars to
    native Python int or float.  All other types are returned unchanged.
    """
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode('utf-8', errors='replace')
    if isinstance(val, np.ndarray):
        if val.dtype.kind in ('S', 'O'):
            return [decode_attr(v) for v in val]
        return val.tolist()
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    return val


def format_num(val, round_digits=None):
    """
    Format a numeric value as a string, rounding to the specified number of
    decimal places when given.  Returns "NaN" for IEEE NaN values and falls
    back to str() for non-numeric input.
    """
    try:
        v = float(val)
        if np.isnan(v):
            return "NaN"
        if round_digits is not None:
            return f"{v:.{round_digits}f}"
        return f"{v:g}"
    except (ValueError, TypeError):
        return str(val)


def extract_vector(val):
    """
    Extract two numeric components from an HDF5 attribute value.

    Accepts a numeric array of length >= 2 (e.g., a [lat, lon] pair stored
    as a float array) or a string containing at least two numeric tokens
    (e.g., "19.5, -65.0" or "19kts, 290deg").  Returns (None, None) if
    fewer than two numbers can be found.
    """
    if val is None:
        return None, None
    if isinstance(val, (list, np.ndarray)):
        arr = list(val)
        if len(arr) >= 2:
            try:
                return float(arr[0]), float(arr[1])
            except (ValueError, TypeError):
                pass
    nums = re.findall(r'[-+]?\d*\.?\d+', str(val))
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    if len(nums) == 1:
        return float(nums[0]), None
    return val, None


def cf_seconds_to_iso(seconds):
    """
    Convert a CF-convention time value (seconds since 1900-01-01 00:00:00Z)
    to an ISO-8601 string.  Returns the raw value as a string if conversion
    fails or the input is NaN.
    """
    try:
        s = float(seconds)
        if np.isnan(s):
            return "NaN"
        dt = _CF_EPOCH + timedelta(seconds=s)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return str(seconds)


def _is_cf_time_dataset(ds):
    """
    Return True if a dataset uses CF-convention time encoding, identified by
    a 'units' attribute containing the string 'seconds since 1900'.
    """
    units = ds.attrs.get('units', b'')
    if isinstance(units, (bytes, np.bytes_)):
        units = units.decode('utf-8', errors='replace')
    return 'seconds since 1900' in str(units).lower()


def _decode_string_list(val):
    """
    Decode an HDF5 attribute that may be stored as a numpy string array, a
    comma-separated string, or a plain string, returning a single display
    string with elements joined by ", ".
    """
    decoded = decode_attr(val)
    if isinstance(decoded, list):
        return ", ".join(str(x) for x in decoded)
    return str(decoded)


# =============================================================================
# DATA EXTRACTION FUNCTIONS
# =============================================================================

def extract_metadata(attrs):
    """
    Extract and format the targeted global metadata fields from an HDF5
    file's root attribute collection, returning a dictionary keyed on the
    field names defined in META_ORDER.

    Storm motion speed and heading are read from their dedicated scalar
    attributes (storm_motion_speed_kt, storm_motion_heading_deg) when
    present, with automatic fallback to parsing the storm_motion string
    attribute for files that store only the combined value.

    The storm center (center_from_tc_vitals) is handled as either a two-
    element float array or a legacy comma-separated string.
    """
    m = {}

    # Storm center coordinates
    tc_raw = attrs.get('center_from_tc_vitals')
    if tc_raw is not None:
        tc_lat, tc_lon = extract_vector(decode_attr(tc_raw))
    else:
        tc_lat, tc_lon = None, None

    m['center_from_tc_vitals_lat'] = format_num(tc_lat, 2) if tc_lat is not None else "N/A"
    m['center_from_tc_vitals_lon'] = format_num(tc_lon, 2) if tc_lon is not None else "N/A"

    # Radius of maximum wind
    rmw = decode_attr(attrs.get('radius_of_maximum_wind_km', ''))
    m['radius_of_maximum_wind_km'] = format_num(rmw, 2) if rmw not in ('', None) else "N/A"

    # Storm cycle date and time, split into component fields
    sdt = decode_attr(attrs.get('storm_datetime', ''))
    if sdt:
        try:
            dt = datetime.strptime(str(sdt).strip(), "%Y-%m-%dT%H:%M:%SZ")
            m['storm_datetime_year']  = str(dt.year)
            m['storm_datetime_month'] = f"{dt.month:02d}"
            m['storm_datetime_day']   = f"{dt.day:02d}"
            m['storm_datetime_hour']  = f"{dt.hour:02d}Z"
        except ValueError:
            s = str(sdt)
            m['storm_datetime_year']  = s[:4]        if len(s) >= 4  else "N/A"
            m['storm_datetime_month'] = s[5:7]        if len(s) >= 7  else "N/A"
            m['storm_datetime_day']   = s[8:10]       if len(s) >= 10 else "N/A"
            m['storm_datetime_hour']  = s[11:13]+"Z"  if len(s) >= 13 else "N/A"
    else:
        for k in ('storm_datetime_year', 'storm_datetime_month',
                  'storm_datetime_day',  'storm_datetime_hour'):
            m[k] = "N/A"

    # Storm identifier
    sid = decode_attr(attrs.get('storm_id', ''))
    m['storm_id'] = str(sid) if sid not in ('', None) else "N/A"

    # Intensity and storm-scale parameters
    sint = decode_attr(attrs.get('storm_intensity_ms', ''))
    m['storm_intensity_ms'] = format_num(sint, 2) if sint not in ('', None) else "N/A"

    mslp = decode_attr(attrs.get('storm_mslp_hpa', ''))
    m['storm_mslp_hpa'] = format_num(mslp) if mslp not in ('', None) else "N/A"

    tcc = decode_attr(attrs.get('tc_category', ''))
    m['tc_category'] = str(tcc) if tcc not in ('', None) else "N/A"

    # Storm motion: read dedicated scalar attributes; fall back to parsing
    # the combined storm_motion string if the scalars are absent or NaN.
    spd_val, hdg_val = None, None

    for attr_name, target in (('storm_motion_speed_kt', 'spd'),
                               ('storm_motion_heading_deg', 'hdg')):
        raw = attrs.get(attr_name)
        if raw is not None:
            try:
                v = float(decode_attr(raw))
                if not np.isnan(v):
                    if target == 'spd':
                        spd_val = v
                    else:
                        hdg_val = v
            except (ValueError, TypeError):
                pass

    if spd_val is None or hdg_val is None:
        sm_raw = attrs.get('storm_motion')
        if sm_raw is not None:
            sm = decode_attr(sm_raw)
            if isinstance(sm, list):
                sm = ', '.join(str(x) for x in sm)
            sm_spd, sm_hdg = extract_vector(sm)
            if spd_val is None and sm_spd is not None:
                spd_val = sm_spd
            if hdg_val is None and sm_hdg is not None:
                hdg_val = sm_hdg

    m['storm_motion_speed_kt']    = format_num(spd_val, 2) if spd_val is not None else "N/A"
    m['storm_motion_heading_deg'] = format_num(hdg_val, 2) if hdg_val is not None else "N/A"

    # Dataset version
    ver = decode_attr(attrs.get('version_number', ''))
    m['version_number'] = str(ver) if ver not in ('', None) else "N/A"

    # Temporal coverage bounds
    tcs = decode_attr(attrs.get('time_coverage_start', ''))
    tce = decode_attr(attrs.get('time_coverage_end',   ''))
    m['time_coverage_start'] = str(tcs) if tcs not in ('', None) else "N/A"
    m['time_coverage_end']   = str(tce) if tce not in ('', None) else "N/A"

    # Observation groups present in this file and the full expected set
    eg_raw = attrs.get('existing_groups')
    m['existing_groups'] = _decode_string_list(eg_raw) if eg_raw is not None else "N/A"

    xg_raw = attrs.get('expected_groups')
    m['expected_groups'] = _decode_string_list(xg_raw) if xg_raw is not None else "N/A"

    return m


def extract_ships(f):
    """
    Extract SHIPS environmental parameters from the ships_params group of an
    open HDF5 file object, returning a dictionary keyed on variable short
    names.  Each entry contains the variable's long name, formatted value,
    and a combined display name suitable for printing.  Returns an empty
    dictionary if the ships_params group is absent.
    """
    ships_data = {}
    if 'ships_params' not in f or not isinstance(f['ships_params'], h5py.Group):
        return ships_data

    ships_grp = f['ships_params']
    ships_vars = [d for d in ships_grp.keys()
                  if isinstance(ships_grp[d], h5py.Dataset)]

    if not ships_vars:
        return ships_data

    try:
        if len(ships_grp[ships_vars[0]]) == 0:
            return ships_data

        for var_name in ships_vars:
            ds  = ships_grp[var_name]
            val = ds[0]

            long_name_raw = ds.attrs.get('long_name')
            long_name = decode_attr(long_name_raw) if long_name_raw is not None else ""
            if isinstance(long_name, list):
                long_name = ' '.join(str(x) for x in long_name)

            display_name = f"{long_name} ({var_name})" if long_name else var_name

            if isinstance(val, (np.floating, float)):
                formatted_val = "NaN" if np.isnan(val) else f"{val:.4g}"
            else:
                formatted_val = decode_attr(val)

            ships_data[var_name] = {
                'long_name':    long_name,
                'value':        formatted_val,
                'display_name': display_name,
            }
    except Exception:
        pass

    return ships_data


def extract_groups(f):
    """
    Inventory all observation groups in an open HDF5 file, returning a
    dictionary keyed on group name.  Each entry records the observation
    count, the number of coordinate variables, and the number of data
    variables.  Groups listed in EXPECTED_GROUPS but absent from the file
    are included with zero counts so that batch comparisons show a consistent
    group universe across files.

    Observation count is read from the obs_count group attribute when
    present, with array length used as a fallback.
    """
    actual_groups = [g for g in f.keys() if isinstance(f[g], h5py.Group)]
    all_groups    = sorted(set(EXPECTED_GROUPS) | set(actual_groups))
    g_data = {}

    for grp_name in all_groups:
        obs_count, c_count, v_count = 0, 0, 0
        exists = False

        if grp_name in f and isinstance(f[grp_name], h5py.Group):
            exists   = True
            grp      = f[grp_name]
            datasets = [d for d in grp.keys()
                        if isinstance(grp[d], h5py.Dataset)]

            attr_count = grp.attrs.get('obs_count', None)
            if attr_count is not None:
                try:
                    obs_count = int(attr_count)
                except (ValueError, TypeError):
                    obs_count = len(grp[datasets[0]]) if datasets else 0
            elif datasets:
                obs_count = len(grp[datasets[0]])

            c_count = sum(1 for d in datasets if d.lower() in KNOWN_COORDS)
            v_count = sum(1 for d in datasets if d.lower() not in KNOWN_COORDS)

        g_data[grp_name] = {
            'obs':     obs_count,
            'c_count': c_count,
            'v_count': v_count,
            'exists':  exists,
        }

    return g_data


# =============================================================================
# CSV EXPORT
# =============================================================================

def _write_hdf5_to_csv_writer(f, fname, writer):
    """
    Write the complete contents of one open HDF5 file to an active CSV
    writer object.  Time arrays stored in CF convention (seconds since
    1900-01-01 00:00:00Z) are converted to ISO-8601 strings automatically.

    Block structure written per file:
      filename row -> metadata rows -> SHIPS rows -> observation group blocks
    Sections are separated by "---" rows; each file ends with "===".
    """
    # Filename
    writer.writerow([fname])

    # Metadata
    writer.writerow(["metadata_param", "metadata_value"])
    m = extract_metadata(f.attrs)
    for key in META_ORDER:
        writer.writerow([key, m.get(key, "N/A")])
    writer.writerow(["---"])

    # SHIPS parameters
    ships = extract_ships(f)
    writer.writerow(["ships_param_short", "ships_param_long", "ships_value"])
    if ships:
        for k, data in sorted(ships.items()):
            safe_long = data['long_name'].strip().replace(" ", "_") \
                if data['long_name'] else ""
            writer.writerow([k, safe_long, data['value']])
    else:
        writer.writerow(["No SHIPS data found", "", ""])
    writer.writerow(["---"])

    # Observation groups
    actual_groups = [g for g in f.keys() if isinstance(f[g], h5py.Group)]
    all_groups    = sorted(set(EXPECTED_GROUPS) | set(actual_groups))

    for grp_name in all_groups:
        if grp_name not in f or not isinstance(f[grp_name], h5py.Group):
            continue

        grp      = f[grp_name]
        datasets = sorted(d for d in grp.keys()
                          if isinstance(grp[d], h5py.Dataset))

        if not datasets:
            continue

        num_rows = len(grp[datasets[0]])
        if num_rows == 0:
            continue

        writer.writerow(["obs_group_name", "number_of_columns", "number_of_rows"])
        writer.writerow([grp_name, len(datasets), num_rows])
        writer.writerow(datasets)

        # Pre-load arrays and flag CF-time columns for ISO-8601 conversion
        arrays   = [grp[d][:] for d in datasets]
        cf_flags = [_is_cf_time_dataset(grp[d]) for d in datasets]

        for i in range(num_rows):
            row_data = []
            for arr, cf_flag in zip(arrays, cf_flags):
                val = arr[i]
                if isinstance(val, bytes):
                    val = val.decode('utf-8', errors='replace')
                elif cf_flag:
                    val = cf_seconds_to_iso(val)
                row_data.append(val)
            writer.writerow(row_data)

        writer.writerow(["---"])

    writer.writerow(["==="])


# =============================================================================
# DISPLAY ROUTINES
# =============================================================================

def process_file_individual(filename, is_compact):
    """
    Print the metadata, SHIPS parameters, and observation-group inventory for
    a single HDF5 file.  When is_compact is True, groups are shown in a
    condensed tabular layout; otherwise each group is printed as a labeled
    block with descriptive annotations.
    """
    print("\n" + "=" * 116)
    print(f"📁 Processing File: {filename}")
    print("=" * 116)

    try:
        with h5py.File(filename, 'r') as f:

            print("\n[ GLOBAL METADATA ]")
            print("-" * 116)
            if not f.attrs:
                print("  (No global metadata found)")
            else:
                m = extract_metadata(f.attrs)
                for key in META_ORDER:
                    val = m.get(key, "N/A")
                    if len(str(val)) > 70:
                        print(f"  {key:<35} :")
                        words = str(val).split(", ")
                        line  = "    "
                        for w in words:
                            if len(line) + len(w) + 2 > 116:
                                print(line.rstrip(", "))
                                line = "    "
                            line += w + ", "
                        if line.strip():
                            print(line.rstrip(", "))
                    else:
                        print(f"  {key:<35} : {val}")

            print("\n[ SHIPS PARAMETERS ]")
            print("-" * 116)
            ships_data = extract_ships(f)
            if ships_data:
                for var_name, data in sorted(ships_data.items()):
                    print(f"  {data['display_name']:<90} : {data['value']}")
            else:
                print("  (No SHIPS data found in this file)")

            g_data = extract_groups(f)
            groups_with_data = sum(1 for d in g_data.values() if d['obs'] > 0)
            print(f"\n\n[ OBSERVATION GROUPS ] "
                  f"({groups_with_data}/{len(g_data)} groups contain data)")

            if is_compact:
                print("=" * 116)
                print(f"{'Group Name':<40} | {'Coords':<15} | {'Vars':<15} | "
                      f"{'Observations':<15}")
                print("-" * 116)
            else:
                print("-" * 30)

            for grp_name, stats in sorted(g_data.items()):
                if is_compact:
                    print(f"{grp_name:<40} | {stats['c_count']:<15} | "
                          f"{stats['v_count']:<15} | {stats['obs']:<15,}")
                else:
                    print(f"\n  🔹 Group: {grp_name}")
                    print(f"     Observations: {stats['obs']:,}")
                    if stats['obs'] > 0:
                        print(f"     Coordinate Variables : {stats['c_count']}")
                        print(f"     Data Variables       : {stats['v_count']}")
                    else:
                        print("     (No data available)")

            if is_compact:
                print("=" * 116)

    except Exception as e:
        print(f"\n❌ An unexpected error occurred while processing {filename}: {e}")


def process_files_tabled(selected_pairs):
    """
    Aggregate metadata, SHIPS parameters, and group statistics for all
    selected files and display them in side-by-side comparison tables.
    """
    print("\n" + "=" * 116)
    print("📊 BATCH PROCESSING: TABLED OVERVIEW")
    print("=" * 116)
    for idx, fname in selected_pairs:
        print(f"  File {idx:<3} : {fname}")

    all_meta, all_ships, all_groups = [], [], []
    all_ships_keys = set()
    total_files    = len(selected_pairs)

    for idx, fname in selected_pairs:
        try:
            with h5py.File(fname, 'r') as f:
                all_meta.append(extract_metadata(f.attrs))
                ships = extract_ships(f)
                all_ships.append(ships)
                all_groups.append(extract_groups(f))
                all_ships_keys.update(ships.keys())
        except Exception as e:
            all_meta.append({k: "ERROR" for k in META_ORDER})
            all_ships.append({})
            all_groups.append({})
            print(f"❌ Error reading File {idx}: {e}")

    file_headers = "".join([f"| File {idx:<10} " for idx, _ in selected_pairs])
    col_line_len = 41 + 18 * len(selected_pairs)

    print(f"\n[ GLOBAL METADATA ]")
    print("-" * col_line_len)
    print(f"{'Parameter':<40} {file_headers}")
    print("-" * col_line_len)
    for k in META_ORDER:
        row = f"{k:<40} "
        for i in range(len(selected_pairs)):
            val = str(all_meta[i].get(k, "N/A"))
            if len(val) > 14:
                val = val[:11] + "..."
            row += f"| {val:<15} "
        print(row)

    print(f"\n[ SHIPS PARAMETERS ]")
    print("-" * col_line_len)
    if not all_ships_keys:
        print("  (No SHIPS data found in any selected files)")
    else:
        print(f"{'Parameter':<40} {file_headers}")
        print("-" * col_line_len)
        for k in sorted(all_ships_keys):
            display_name = next(
                (all_ships[i][k]['display_name']
                 for i in range(len(selected_pairs))
                 if k in all_ships[i]),
                k,
            )
            display_k = display_name if len(display_name) <= 39 \
                else display_name[:36] + "..."
            row = f"{display_k:<40} "
            for i in range(len(selected_pairs)):
                val = all_ships[i].get(k, {}).get('value', "N/A")
                row += f"| {val:<15} "
            print(row)

    print(f"\n[ OBSERVATION GROUPS ] (Aggregated across {total_files} files)")
    obs_line_len = 112
    print("-" * obs_line_len)
    print(f"{'Group Name':<40} | {'Max Coords':<15} | {'Max Vars':<15} | "
          f"{'Total Obs':<15} | {'Average Obs':<15}")
    print("-" * obs_line_len)

    universe_groups = set(EXPECTED_GROUPS)
    for g_dict in all_groups:
        universe_groups.update(g_dict.keys())

    for grp_name in sorted(universe_groups):
        max_c, max_v, tot_obs, file_count = 0, 0, 0, 0
        for g_dict in all_groups:
            if grp_name in g_dict:
                d = g_dict[grp_name]
                max_c    = max(max_c, d['c_count'])
                max_v    = max(max_v, d['v_count'])
                tot_obs += d['obs']
                if d.get('exists', False):
                    file_count += 1

        avg_obs   = tot_obs / total_files if total_files > 0 else 0
        grp_label = f"{grp_name} ({file_count}/{total_files} files)"
        if len(grp_label) > 40:
            grp_label = grp_label[:37] + "..."
        print(f"{grp_label:<40} | {max_c:<15} | {max_v:<15} | "
              f"{tot_obs:<15,} | {avg_obs:<15,.0f}")

    print("=" * obs_line_len)


def process_files_to_single_csv(selected_pairs,
                                output_filename="hrdobs_file_contents.csv"):
    """
    Export the complete contents of all selected files into a single CSV
    file.  See the module docstring for a description of the output structure.
    """
    print("\n" + "=" * 116)
    print(f"💾 EXPORTING FULL DATA CONTENTS TO SINGLE CSV: {output_filename}")
    print("=" * 116)

    with open(output_filename, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        for idx, fname in selected_pairs:
            print(f"  Exporting File {idx:<3} : {fname}")
            try:
                with h5py.File(fname, 'r') as f:
                    _write_hdf5_to_csv_writer(f, fname, writer)
            except Exception as e:
                print(f"❌ Error processing {fname}: {e}")

    print(f"\n✅ Data exported to {output_filename}")


def process_files_to_individual_csvs(selected_pairs):
    """
    Export the complete contents of each selected file to its own CSV file,
    named hrdobs_file_contents_N.csv where N is the file's selection index.
    See the module docstring for a description of the output structure.
    """
    print("\n" + "=" * 116)
    print("💾 EXPORTING FULL DATA CONTENTS TO INDIVIDUAL CSV FILES")
    print("=" * 116)

    for idx, fname in selected_pairs:
        out_name = f"hrdobs_file_contents_{idx}.csv"
        print(f"  Exporting File {idx:<3} : {fname} -> {out_name}")
        try:
            with open(out_name, 'w', newline='', encoding='utf-8') as f_out:
                writer = csv.writer(f_out)
                with h5py.File(fname, 'r') as f:
                    _write_hdf5_to_csv_writer(f, fname, writer)
        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")

    print("\n✅ All files exported successfully.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 116)
    print("📊 HRDOBS AI-Ready Dataset Reader & Converter  (v1.0)")
    print("=" * 116)

    # Scan for HDF5 files in the current directory
    hdf5_files = sorted(glob.glob("*.hdf5"))

    if not hdf5_files:
        print("\n❌ No .hdf5 files found in the current directory.")
        print("Please ensure this script is run from the folder containing "
              "your HRDOBS .hdf5 files.")
        return

    print("\nFound the following HDF5 files:")
    for i, file_name in enumerate(hdf5_files, 1):
        print(f"  {i}. {file_name}")

    # File selection
    selection = input(
        "\nSelect files to process (0 = all, e.g. 1, 3-5, 8): "
    ).strip()

    selected_pairs = []
    if selection == '0':
        selected_pairs = [(i + 1, fname) for i, fname in enumerate(hdf5_files)]
    else:
        try:
            indices = set()
            for part in selection.split(","):
                part = part.strip()
                if "-" in part:
                    start, end = map(int, part.split("-"))
                    if start > end:
                        start, end = end, start
                    indices.update(range(start, end + 1))
                else:
                    indices.add(int(part))
            selected_pairs = [
                (i, hdf5_files[i - 1])
                for i in sorted(indices)
                if 1 <= i <= len(hdf5_files)
            ]
        except ValueError:
            print("\n❌ Invalid selection. Use numbers, ranges (e.g. 2-5), "
                  "or combinations (e.g. 1, 3-5, 8).")
            return

    if not selected_pairs:
        print("\n❌ No valid files were selected.")
        return

    # Output format selection
    print("\nSelect an output format:")
    print("  1  Expanded listing       — full detail, one file at a time")
    print("  2  Compact listing        — condensed table, one file at a time")
    print("  3  Tabled batch overview  — side-by-side comparison across files")
    print("  4  Single CSV export      — all files to hrdobs_file_contents.csv")
    print("  5  Individual CSV export  — one CSV per file")

    choice = input("\nEnter choice (1-5) [default: 1]: ").strip()
    if choice not in ('1', '2', '3', '4', '5'):
        choice = '1'

    if choice in ('1', '2'):
        for idx, filename in selected_pairs:
            process_file_individual(filename, is_compact=(choice == '2'))
    elif choice == '3':
        process_files_tabled(selected_pairs)
    elif choice == '4':
        process_files_to_single_csv(selected_pairs)
    elif choice == '5':
        process_files_to_individual_csvs(selected_pairs)

    print("\n✅ Processing complete.\n")


if __name__ == "__main__":
    main()
