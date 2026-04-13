#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hrdobs_sample_reader.py
-----------------------
A standalone companion script to read and summarize AI-Ready HRDOBS HDF5 files.
This script scans the directory, allows users to select files to process, 
extracts targeted global metadata, lists SHIPS environmental parameters, and provides 
a structural inventory of the observation groups contained within.

Output options include expanded lists, compact lists, an aggregated tabled overview,
or exporting the FULL dataset contents (metadata + all data arrays) to structural CSVs.
"""

import h5py
import numpy as np
import glob
import re
import csv
import os
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================
# Common base coordinates found in HRDOBS files
KNOWN_COORDS = {
    'time', 'lat', 'lon', 'latitude', 'longitude', 'clat', 'clon', 
    'height', 'altitude', 'elev', 'p', 'pres', 'pressure', 'ght', 'rmw'
}

# The universe of expected observation groups across the entire HRDOBS dataset
EXPECTED_GROUPS = [
    "dropsonde_ghawk", "dropsonde_noaa42", "dropsonde_noaa43", "dropsonde_noaa49", "dropsonde_usaf",
    "flight_level_hdobs_noaa42", "flight_level_hdobs_noaa43", "flight_level_hdobs_noaa49", "flight_level_hdobs_usaf",
    "sfmr_noaa42", "sfmr_noaa43", "sfmr_usaf",
    "tdr_noaa42", "tdr_noaa43", "tdr_noaa49",
    "ships_params",
    "track_best_track", "track_spline_track", "track_vortex_message"
]

META_ORDER = [
    'center_from_tc_vitals_lat', 'center_from_tc_vitals_lon', 'radius_of_maximum_wind_km', 
    'storm_datetime_year', 'storm_datetime_month', 'storm_datetime_day', 'storm_datetime_hour', 
    'storm_id', 'storm_intensity_ms', 'storm_motion_speed', 
    'storm_motion_dir', 'storm_mslp_hpa', 'tc_category'
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def decode_attr(val):
    """Safely decodes HDF5 byte attributes to strings for printing."""
    if isinstance(val, bytes):
        return val.decode('utf-8', errors='replace')
    elif isinstance(val, np.ndarray) and val.dtype.kind in ('S', 'O'):
        return [decode_attr(v) for v in val]
    return val

def format_num(val, round_digits=None):
    """Safely formats and rounds numeric values."""
    try:
        v = float(val)
        if round_digits is not None:
            return f"{v:.{round_digits}f}"
        return f"{v:g}"
    except (ValueError, TypeError):
        return str(val)

def extract_vector(val):
    """Extracts exactly two numeric components from an HDF5 array or string."""
    if val is None:
        return None, None
    if isinstance(val, (list, np.ndarray)) and len(val) >= 2:
        return float(val[0]), float(val[1])
    nums = re.findall(r'[-+]?\d*\.?\d+', str(val))
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return val, None

# --- Data Extraction Helpers for Tabled Processing ---

def extract_metadata(attrs):
    """Extracts and formats targeted metadata into a clean dictionary."""
    m = {}
    tc_lat, tc_lon = extract_vector(decode_attr(attrs.get('center_from_tc_vitals')))
    m['center_from_tc_vitals_lat'] = format_num(tc_lat, 2) if tc_lat is not None else "N/A"
    m['center_from_tc_vitals_lon'] = format_num(tc_lon, 2) if tc_lon is not None else "N/A"
    
    rmw = decode_attr(attrs.get('radius_of_maximum_wind_km'))
    m['radius_of_maximum_wind_km'] = format_num(rmw, 2) if rmw is not None else "N/A"
    
    # Safely parse and split the storm_datetime string
    sdt = decode_attr(attrs.get('storm_datetime'))
    if sdt:
        try:
            dt = datetime.strptime(sdt.strip(), "%Y-%m-%dT%H:%M:%SZ")
            m['storm_datetime_year'] = str(dt.year)
            m['storm_datetime_month'] = f"{dt.month:02d}"
            m['storm_datetime_day'] = f"{dt.day:02d}"
            m['storm_datetime_hour'] = f"{dt.hour:02d}Z"
        except ValueError:
            m['storm_datetime_year'] = sdt[:4] if len(sdt) >= 4 else "N/A"
            m['storm_datetime_month'] = sdt[5:7] if len(sdt) >= 7 else "N/A"
            m['storm_datetime_day'] = sdt[8:10] if len(sdt) >= 10 else "N/A"
            m['storm_datetime_hour'] = sdt[11:13] + "Z" if len(sdt) >= 13 else "N/A"
    else:
        m['storm_datetime_year'] = "N/A"
        m['storm_datetime_month'] = "N/A"
        m['storm_datetime_day'] = "N/A"
        m['storm_datetime_hour'] = "N/A"

    sid = decode_attr(attrs.get('storm_id'))
    m['storm_id'] = sid if sid is not None else "N/A"
    
    sint = decode_attr(attrs.get('storm_intensity_ms'))
    m['storm_intensity_ms'] = format_num(sint, 2) if sint is not None else "N/A"
    
    sm_spd, sm_dir = extract_vector(decode_attr(attrs.get('storm_motion')))
    m['storm_motion_speed'] = format_num(sm_spd, 2) if sm_spd is not None else "N/A"
    m['storm_motion_dir'] = format_num(sm_dir, 2) if sm_dir is not None else "N/A"
    
    mslp = decode_attr(attrs.get('storm_mslp_hpa'))
    m['storm_mslp_hpa'] = format_num(mslp) if mslp is not None else "N/A"
    
    tcc = decode_attr(attrs.get('tc_category'))
    m['tc_category'] = tcc if tcc is not None else "N/A"
    
    return m

def extract_ships(f):
    """Extracts SHIPS parameters and returns a structured dictionary for easy splitting."""
    ships_data = {}
    if 'ships_params' in f and isinstance(f['ships_params'], h5py.Group):
        ships_grp = f['ships_params']
        ships_vars = [d for d in ships_grp.keys() if isinstance(ships_grp[d], h5py.Dataset)]
        if ships_vars:
            try:
                if len(ships_grp[ships_vars[0]]) > 0:
                    for var_name in ships_vars:
                        ds = ships_grp[var_name]
                        val = ds[0]
                        long_name_raw = ds.attrs.get('long_name')
                        long_name = decode_attr(long_name_raw) if long_name_raw is not None else ""
                        display_name = f"{long_name} ({var_name})" if long_name else var_name
                        
                        if isinstance(val, (np.floating, float)):
                            formatted_val = f"{val:.4g}"
                        else:
                            formatted_val = decode_attr(val)
                            
                        ships_data[var_name] = {
                            'long_name': long_name,
                            'value': formatted_val,
                            'display_name': display_name
                        }
            except Exception:
                pass
    return ships_data

def extract_groups(f):
    """Inventories observation groups, yielding counts of observations, coords, and vars."""
    actual_groups = [g for g in f.keys() if isinstance(f[g], h5py.Group)]
    all_groups = sorted(list(set(EXPECTED_GROUPS + actual_groups)))
    g_data = {}
    
    for grp_name in all_groups:
        obs_count, c_count, v_count = 0, 0, 0
        exists = False
        if grp_name in f and isinstance(f[grp_name], h5py.Group):
            exists = True
            grp = f[grp_name]
            datasets = [d for d in grp.keys() if isinstance(grp[d], h5py.Dataset)]
            if datasets:
                obs_count = len(grp[datasets[0]])
            c_count = len([d for d in datasets if d.lower() in KNOWN_COORDS])
            v_count = len([d for d in datasets if d.lower() not in KNOWN_COORDS])
        
        g_data[grp_name] = {'obs': obs_count, 'c_count': c_count, 'v_count': v_count, 'exists': exists}
    return g_data

# =============================================================================
# CSV EXPORT HELPER
# =============================================================================

def _write_hdf5_to_csv_writer(f, fname, writer):
    """Core logic to extract and write a single HDF5 file's contents to an active CSV writer."""
    # 1. Filename row
    writer.writerow([fname])
    
    # 2. Main 2-column Header for metadata
    writer.writerow(["metadata_param", "metadata_value"])
    
    # 3. Write Metadata
    m = extract_metadata(f.attrs)
    for key in META_ORDER:
        writer.writerow([key, m.get(key, "N/A")])
        
    writer.writerow(["---"]) # Spacer after metadata
        
    # 4. Write SHIPS
    ships = extract_ships(f)
    if ships:
        writer.writerow(["ships_param_short", "ships_param_long", "ships_value"]) # SHIPS Header
        for k, data in sorted(ships.items()):
            # Replace spaces with underscores for robust parsing in scientific software
            safe_long_name = data['long_name'].strip().replace(" ", "_") if data['long_name'] else ""
            writer.writerow([k, safe_long_name, data['value']])
    else:
        writer.writerow(["ships_param_short", "ships_param_long", "ships_value"])
        writer.writerow(["No SHIPS data found", "", ""])
            
    writer.writerow(["---"]) # Spacer after SHIPS / before groups
    
    # 5. Extract Data from Observation Groups
    actual_groups = [g for g in f.keys() if isinstance(f[g], h5py.Group)]
    all_groups = sorted(list(set(EXPECTED_GROUPS + actual_groups)))
    
    for grp_name in all_groups:
        if grp_name in f and isinstance(f[grp_name], h5py.Group):
            grp = f[grp_name]
            datasets = sorted([d for d in grp.keys() if isinstance(grp[d], h5py.Dataset)])
            
            if datasets:
                num_rows = len(grp[datasets[0]])
                num_cols = len(datasets)
                
                if num_rows > 0:
                    # Write Group Schema Header
                    writer.writerow(["obs_group_name", "number_of_columns", "number_of_rows"])
                    writer.writerow([grp_name, num_cols, num_rows])
                    
                    # Write Dataset Names (Column Headers)
                    writer.writerow(datasets)
                    
                    # Pre-load all 1D data arrays into memory for fast writing
                    arrays = [grp[d][:] for d in datasets]
                    
                    # Write actual data rows
                    for i in range(num_rows):
                        row_data = []
                        for arr in arrays:
                            val = arr[i]
                            # Decode byte strings natively if they exist
                            if isinstance(val, bytes):
                                val = val.decode('utf-8', errors='replace')
                            row_data.append(val)
                        writer.writerow(row_data)
                        
                    writer.writerow(["---"]) # Spacer after each observation group
                    
    writer.writerow(["==="]) # Extra spacer marking the end of the file

# =============================================================================
# PROCESSING ROUTINES
# =============================================================================

def process_file_individual(filename, is_compact):
    """Processes and prints the contents of a single HDF5 file sequentially."""
    print("\n" + "=" * 116)
    print(f"📁 Processing File: {filename}")
    print("=" * 116)
    
    try:
        with h5py.File(filename, 'r') as f:
            
            # 1. METADATA
            print("\n[ GLOBAL METADATA ]")
            print("-" * 116)
            if not f.attrs:
                print("  (No global metadata found)")
            else:
                m = extract_metadata(f.attrs)
                for key in META_ORDER:
                    print(f"  {key:<35} : {m[key]}")
            
            # 2. SHIPS PARAMETERS
            print("\n[ SHIPS PARAMETERS ]")
            print("-" * 116)
            ships_data = extract_ships(f)
            if ships_data:
                for var_name, data in sorted(ships_data.items()):
                    print(f"  {data['display_name']:<90} : {data['value']}")
            else:
                print("  (No SHIPS data found in this file)")

            # 3. OBSERVATION GROUPS
            g_data = extract_groups(f)
            groups_with_data = sum(1 for d in g_data.values() if d['obs'] > 0)
            
            print(f"\n\n[ OBSERVATION GROUPS ] ({groups_with_data}/{len(g_data)} groups contain data)")
            
            if is_compact:
                print("=" * 116)
                print(f"{'Group Name':<40} | {'Coords':<15} | {'Vars':<15} | {'Observations':<15}")
                print("-" * 116)
            else:
                print("-" * 30)
            
            for grp_name, stats in sorted(g_data.items()):
                if is_compact:
                    print(f"{grp_name:<40} | {stats['c_count']:<15} | {stats['v_count']:<15} | {stats['obs']:<15,}")
                else:
                    print(f"\n  🔹 Group: {grp_name}")
                    print(f"     Observations: {stats['obs']:,}")
                    if stats['obs'] > 0:
                        print(f"     Coordinate Vars: {stats['c_count']}")
                        print(f"     Data Variables : {stats['v_count']}")
                    else:
                        print("     (No data available)")
                        
            if is_compact:
                print("=" * 116)
                
    except Exception as e:
        print(f"\n❌ An unexpected error occurred while processing {filename}: {e}")

def process_files_tabled(selected_pairs):
    """Processes multiple HDF5 files and aggregates their contents into dynamic tables."""
    print("\n" + "=" * 116)
    print("📊 BATCH PROCESSING: TABLED OVERVIEW")
    print("=" * 116)
    for idx, fname in selected_pairs:
        print(f"  File {idx:<3} : {fname}")
    
    all_meta, all_ships, all_groups = [], [], []
    all_ships_keys = set()
    total_files = len(selected_pairs)
    
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

    file_headers = "".join([f"| File {idx:<10} " for idx, fname in selected_pairs])
    col_line_len = 41 + 18 * len(selected_pairs)
    
    # Metadata Table
    print(f"\n[ GLOBAL METADATA ]")
    print("-" * col_line_len)
    print(f"{'Parameter':<40} {file_headers}")
    print("-" * col_line_len)
    for k in META_ORDER:
        row = f"{k:<40} "
        for i in range(len(selected_pairs)):
            val = all_meta[i].get(k, "N/A")
            row += f"| {val:<15} "
        print(row)
        
    # SHIPS Table
    print(f"\n[ SHIPS PARAMETERS ]")
    print("-" * col_line_len)
    if not all_ships_keys:
        print("  (No SHIPS data found in any selected files)")
    else:
        print(f"{'Parameter':<40} {file_headers}")
        print("-" * col_line_len)
        for k in sorted(list(all_ships_keys)):
            display_name = next((all_ships[i][k]['display_name'] for i in range(len(selected_pairs)) if k in all_ships[i]), k)
            display_k = display_name if len(display_name) <= 39 else display_name[:36] + "..."
            row = f"{display_k:<40} "
            for i in range(len(selected_pairs)):
                val = all_ships[i].get(k, {}).get('value', "N/A")
                row += f"| {val:<15} "
            print(row)
            
    # Observation Groups Table
    print(f"\n[ OBSERVATION GROUPS ] (Aggregated across {total_files} files)")
    obs_line_len = 112
    print("-" * obs_line_len)
    print(f"{'Group Name':<40} | {'Max Coords':<15} | {'Max Vars':<15} | {'Total Obs':<15} | {'Average Obs':<15}")
    print("-" * obs_line_len)
    
    universe_groups = set(EXPECTED_GROUPS)
    for g_dict in all_groups:
        universe_groups.update(g_dict.keys())
        
    for grp_name in sorted(list(universe_groups)):
        max_c, max_v, tot_obs, file_count = 0, 0, 0, 0
        for g_dict in all_groups:
            if grp_name in g_dict:
                d = g_dict[grp_name]
                max_c = max(max_c, d['c_count'])
                max_v = max(max_v, d['v_count'])
                tot_obs += d['obs']
                if d.get('exists', False):
                    file_count += 1
                    
        avg_obs = tot_obs / total_files if total_files > 0 else 0
        
        grp_label = f"{grp_name} ({file_count}/{total_files} files)"
        if len(grp_label) > 40:
            grp_label = grp_label[:37] + "..."
            
        print(f"{grp_label:<40} | {max_c:<15} | {max_v:<15} | {tot_obs:<15,} | {avg_obs:<15,.0f}")
    print("=" * obs_line_len)

def process_files_to_single_csv(selected_pairs, output_filename="hrdobs_file_contents.csv"):
    """Exports the FULL dataset contents of ALL selected files to a SINGLE CSV."""
    print("\n" + "=" * 116)
    print(f"💾 BATCH PROCESSING: EXPORTING FULL DATA CONTENTS TO SINGLE CSV -> {output_filename}")
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
                
    print(f"\n✅ Full batch data successfully exported to {output_filename}")

def process_files_to_individual_csvs(selected_pairs):
    """Exports the FULL dataset contents of EACH selected file to its OWN CSV 
    using the selection index in the filename."""
    print("\n" + "=" * 116)
    print(f"💾 BATCH PROCESSING: EXPORTING FULL DATA CONTENTS TO INDIVIDUAL CSVs")
    print("=" * 116)
    
    for idx, fname in selected_pairs:
        # Generate the output filename using the file number (XYZ)
        out_name = f"hrdobs_file_contents_{idx}.csv"
            
        print(f"  Exporting File {idx:<3} : {fname} -> {out_name}")
        
        try:
            with open(out_name, 'w', newline='', encoding='utf-8') as f_out:
                writer = csv.writer(f_out)
                with h5py.File(fname, 'r') as f:
                    _write_hdf5_to_csv_writer(f, fname, writer)
        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")
            
    print(f"\n✅ All individual files successfully exported.")

# =============================================================================
# MAIN INTERACTIVE FUNCTION
# =============================================================================
def main():
    print("=" * 116)
    print(f"📊 HRDOBS AI-Ready Data Reader & Converter")
    print("=" * 116)
    
    # 1. Scan for HDF5 files in the current directory
    hdf5_files = sorted(glob.glob("*.hdf5"))
    
    if not hdf5_files:
        print("\n❌ No .hdf5 files found in the current directory.")
        print("Please ensure this script is located in the same folder as your HRDOBS data.")
        return
        
    print("\nFound the following HDF5 files:")
    for i, file_name in enumerate(hdf5_files, 1):
        print(f"  {i}. {file_name}")
        
    # 2. Prompt user for file selection
    selection = input("\nPlease select files to process (0 for all, e.g., 1, 3-5, 8): ").strip()
    
    selected_pairs = []
    if selection == '0':
        selected_pairs = [(i+1, fname) for i, fname in enumerate(hdf5_files)]
    else:
        try:
            indices = set()
            for part in selection.split(","):
                part = part.strip()
                if "-" in part:
                    start, end = map(int, part.split("-"))
                    # Ensure start <= end just in case user types backwards
                    if start > end:
                        start, end = end, start
                    indices.update(range(start, end + 1))
                else:
                    indices.add(int(part))
            
            # Sort indices and match to files (1-based to 0-based index)
            sorted_indices = sorted(list(indices))
            selected_pairs = [(i, hdf5_files[i-1]) for i in sorted_indices if 1 <= i <= len(hdf5_files)]
        except ValueError:
            print("\n❌ Invalid input. Please enter numbers, ranges separated by hyphens, or commas (e.g., 1, 3-5, 8).")
            return
            
    if not selected_pairs:
        print("\n❌ No valid files were selected.")
        return

    # 3. Prompt user for output format
    print("\nPlease select an output format:")
    print("  1. List of contents (expanded, individual files)")
    print("  2. List of contents (compact, individual files)")
    print("  3. List of contents (compact, tabled batch overview)")
    print("  4. Convert full dataset contents to single CSV file (hrdobs_file_contents.csv)")
    print("  5. Convert full dataset contents to individual CSV files (hrdobs_file_contents_XYZ.csv)")
    
    choice = input("\nEnter choice (1, 2, 3, 4, or 5) [default: 1]: ").strip()
    if choice not in ['1', '2', '3', '4', '5']:
        choice = '1'
    
    # 4. Route to the appropriate processing function
    if choice in ['1', '2']:
        is_compact = (choice == '2')
        for idx, filename in selected_pairs:
            process_file_individual(filename, is_compact)
    elif choice == '3':
        process_files_tabled(selected_pairs)
    elif choice == '4':
        process_files_to_single_csv(selected_pairs)
    elif choice == '5':
        process_files_to_individual_csvs(selected_pairs)
        
    print("\n✅ Processing complete.\n")

if __name__ == "__main__":
    main()
