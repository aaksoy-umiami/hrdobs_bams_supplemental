import os
import glob
import re
import datetime
import h5py
import numpy as np
import pandas as pd
import ujson

# =============================================================================
# CONFIGURATION — must stay in sync with make_ai_ready_batch_v10.py
# =============================================================================

CRITICAL_METADATA = [
    'storm_name', 'storm_id', 'storm_datetime', 'storm_epoch',
    'center_from_tc_vitals', 'tc_category',
    'storm_intensity_ms', 'storm_mslp_hpa', 'radius_of_maximum_wind_km',
]

EXPECTED_METADATA = [
    'creator_email', 'creator_name',
    'Virtual_Manifest',
    'geospatial_lat_max', 'geospatial_lat_min',
    'geospatial_lat_units',
    'geospatial_lon_max', 'geospatial_lon_min',
    'geospatial_lon_units',
    'existing_groups', 'expected_groups', 'storm_motion_speed_kt', 'storm_motion_heading_deg',
    'time_coverage_start', 'time_coverage_end',
    'title', 'version_number',
]

ALL_REQUIRED_ATTRS = CRITICAL_METADATA + EXPECTED_METADATA

BANNED_INSTRUMENTS = ['lidar', 'dwl', 'coyote']

# Sentinel values that must NOT remain in any numeric dataset after conversion.
SENTINEL_VALUES = [-99.0, -999.0, 99.0, 999.0, 9999.0,
                   513.93, 5143.93, -513.93, -5143.93, -9999.0]

# SHIPS uses 99.0 (e.g., shear direction, distance) and 999.0 (e.g., pressure) 
# as valid physical measurements. We exclude them from the SHIPS sentinel checks.
SHIPS_SENTINEL_VALUES = [v for v in SENTINEL_VALUES if v not in (99.0, 999.0)]

VALID_BOUNDS = {
    'lat':       (-90.0,    90.0),
    'latitude':  (-90.0,    90.0),
    'lon':       (-180.0,  180.0),
    'longitude': (-180.0,  180.0),
    'time':      (2839968000.0, 4102444800.0), # Seconds since 1900-01-01 00:00:00Z
    'sfcp':      (80000.0, 110000.0),          # Pa
    'height':    (0.0, 30000.0),               # m
    'altitude':  (0.0, 30000.0),               # m
    'ght':       (0.0, 30000.0),               # m
}

OBS_PROXIMITY_DEG = 10.0
LOCATION_VARS = {'lat', 'latitude', 'lon', 'longitude', 'height', 'altitude', 'ght'}

# =============================================================================
# BASIN FILTERING
# =============================================================================
BASIN_FILTER = 'ATL'

_BASIN_SUFFIX_MAP = {
    'L': 'ATL',
    'E': 'EPAC',
}
_BASIN_LABEL = {'ATL': 'Atlantic', 'EPAC': 'East Pacific', 'ALL': 'All Basins'}

# =============================================================================
# SPLINE TRACK PRESSURE VALIDATION & CATEGORY
# =============================================================================
SPLINE_PRES_BOUNDS = (400.0, 1100.0)

VALID_TC_CATEGORIES = {
    'TD', 'TS', 'H1', 'H2', 'H3', 'H4', 'H5',
    'SS', 'SD', 'EX', 'LO', 'DB', 'WV', 'HU', 'NaN',
}

_NHC_SPELLED_NUMBERS = {
    'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE',
    'TEN', 'ELEVEN', 'TWELVE', 'THIRTEEN', 'FOURTEEN', 'FIFTEEN', 'SIXTEEN',
    'SEVENTEEN', 'EIGHTEEN', 'NINETEEN', 'TWENTY', 'TWENTYONE', 'TWENTYTWO',
    'TWENTYTHREE', 'TWENTYFOUR', 'TWENTYFIVE',
}
_NHC_INVEST       = {'INVEST'}
_NHC_NUMBER_WORDS = _NHC_SPELLED_NUMBERS | _NHC_INVEST

COORD_VARS = {
    'time', 'lat', 'latitude', 'lon', 'longitude',
    'ght', 'p', 'az', 'elev', 'clat', 'clon', 'rr', 
}

TRACK_PRODUCT_VARS = {'vmax', 'pmin', 'rmw', 'sfcp', 'sfcspd', 'sfcdir', 'pres'}

TRACK_GROUP_PREFIX = 'track_'
SPLINE_TRACK_GROUP = 'track_spline_track'

# =============================================================================
# SHIPS PARAMETERS GROUP (v9/v10)
# =============================================================================
SHIPS_GROUP_NAME = 'ships_params'
SHIPS_REQUIRED_ATTRS = {'obs_count', 'source', 'ships_atcf_id', 'ships_datetime_utc'}
SHIPS_EXPECTED_DATASETS = {
    'type', 'incv_kt', 'csst_degc', 'cd20_m', 'cd26_m', 'cohc_kjcm2',
    'dtl_km', 'oage_hr', 'nage_hr', 'shrd_kt', 'shtd_deg', 'shdc_kt',
    'sddc_deg', 'rhlo_pct', 'rhmd_pct', 'rhhi_pct', 'vmpi_kt',
    'penv_hpa', 'penc_hpa', 'z850_1e7_per_s', 'd200_1e7_per_s', 'u200_kt',
    'dsst_degc', 'nsst_degc', 'nohc_kjcm2',
}

# =============================================================================
# HELPERS
# =============================================================================

def decode_attr(val):
    if isinstance(val, (np.ndarray, list)):
        if len(val) == 0: return ''
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
    return str(val).rstrip('\x00').strip()

def is_track_group(group_name): return group_name.lower().startswith(TRACK_GROUP_PREFIX)
def is_ships_group(group_name): return group_name.lower() == SHIPS_GROUP_NAME
def is_coord(var_name): return var_name.lower() in COORD_VARS
def is_track_product(var_name): return var_name.lower() in TRACK_PRODUCT_VARS
def is_err_field(var_name): return var_name.lower().endswith('err')

def detect_basin(storm_id):
    m = re.search(r'\d+([A-Z]+)$', storm_id.upper())
    if m: return _BASIN_SUFFIX_MAP.get(m.group(1), 'UNKNOWN')
    return 'UNKNOWN'

def should_validate_file(filepath):
    fname = os.path.basename(filepath)
    m = re.search(r'HRDOBS_([A-Z0-9]+)[._]', fname)
    if not m: return True
    storm_id = m.group(1)
    basin = detect_basin(storm_id)
    if BASIN_FILTER == 'ALL': return True
    return basin == BASIN_FILTER

def is_nan_value(val):
    if isinstance(val, (float, np.floating)) and np.isnan(val): return True
    if isinstance(val, (str, bytes, np.bytes_)):
        if decode_attr(val).strip().lower() in ('nan', ''): return True
    return False

def validate_json_sidecar(json_path, expected_h5_name):
    """
    Checks if the JSON manifest exists and has a valid Kerchunk structure.
    """
    if not os.path.exists(json_path):
        return f"Missing JSON sidecar manifest."
    
    try:
        with open(json_path, 'r') as f:
            manifest = ujson.load(f)
            
        # Ensure it contains the standard Zarr/Kerchunk dictionary keys
        if 'refs' not in manifest or 'version' not in manifest:
            return "JSON sidecar exists but is missing valid Kerchunk 'refs' or 'version' keys."
            
    except Exception as e:
        return f"JSON sidecar is corrupt or unparseable: {e}"
        
    return None

# =============================================================================
# PER-FILE VALIDATION
# =============================================================================

def validate_file(filepath):
    issues  = []
    filename = os.path.basename(filepath) # 
    
    # --- V10: Check HDF5/JSON Pair Compatibility ---
    json_pair = filepath.replace('.hdf5', '.json')
    pair_issue = validate_json_sidecar(json_pair, filename)
    if pair_issue:
        issues.append(f"[PAIR] {pair_issue}")
    # -----------------------------------------------

    summary = {
        'filename':   os.path.basename(filepath),
        'groups':     [],
        'obs_counts': {},
    }

    try:
        with h5py.File(filepath, 'r') as f:

            # ------------------------------------------------------------------
            # CHECK 1a: Critical metadata (must exist, non-empty, non-NaN, correct type)
            # ------------------------------------------------------------------
            for attr in CRITICAL_METADATA:
                if attr not in f.attrs:
                    issues.append(f"[GLOBAL] Missing critical attribute: '{attr}'")
                    continue

                raw_val = f.attrs[attr]

                if is_nan_value(raw_val):
                    issues.append(
                        f"[GLOBAL] Critical attribute '{attr}' is 'NaN' or empty "
                        f"— file should have been excluded during conversion."
                    )
                    continue

                # Per-attribute format/type validation
                if attr == 'storm_name':
                    val = decode_attr(raw_val)
                    if not re.match(r'^[A-Z]+$', val):
                        issues.append(f"[GLOBAL] '{attr}' invalid name: '{val}'")

                elif attr == 'storm_id':
                    val = decode_attr(raw_val)
                    if not re.match(r'^[A-Z]+\d+[A-Z]+$', val):
                        issues.append(f"[GLOBAL] '{attr}' invalid format: '{val}'")

                elif attr == 'storm_datetime':
                    val = decode_attr(raw_val)
                    try:
                        datetime.datetime.strptime(val, "%Y-%m-%dT%H:%M:%SZ")
                    except ValueError:
                        issues.append(f"[GLOBAL] '{attr}' invalid ISO format: '{val}'")

                elif attr == 'storm_epoch':
                    if not isinstance(raw_val, (int, np.integer)):
                        issues.append(f"[GLOBAL] '{attr}' must be a native integer. Got {type(raw_val)}")
                    else:
                        if raw_val < 631152000 or raw_val > 2051222400:
                            issues.append(f"[GLOBAL] '{attr}' value {raw_val} outside reasonable range.")

                elif attr == 'tc_category':
                    pass  # token validated in CHECK 14

                elif attr in ('storm_intensity_ms', 'storm_mslp_hpa', 'radius_of_maximum_wind_km'):
                    if not isinstance(raw_val, (float, np.floating, int, np.integer)):
                        issues.append(f"[GLOBAL] '{attr}' must be a native numerical type. Got {type(raw_val)}")
                    else:
                        fv = float(raw_val)
                        if attr == 'storm_intensity_ms' and (fv < 0 or fv > 100):
                            issues.append(f"[GLOBAL] '{attr}' value {fv} m/s outside reasonable range.")
                        elif attr == 'storm_mslp_hpa' and (fv < 850 or fv > 1050):
                            issues.append(f"[GLOBAL] '{attr}' value {fv} hPa outside reasonable range.")
                        elif attr == 'radius_of_maximum_wind_km' and (fv < 0 or fv > 750):
                            issues.append(f"[GLOBAL] '{attr}' value {fv} km outside reasonable range.")

                elif attr == 'center_from_tc_vitals':
                    if not isinstance(raw_val, (np.ndarray, list)) or len(raw_val) != 2:
                        issues.append(f"[GLOBAL] '{attr}' must be a 2-element native array [lat, lon].")
                    else:
                        clat, clon = float(raw_val[0]), float(raw_val[1])
                        if clat < -90 or clat > 90:
                            issues.append(f"[GLOBAL] '{attr}' latitude {clat} outside bounds.")
                        if clon < -180 or clon > 180:
                            issues.append(f"[GLOBAL] '{attr}' longitude {clon} outside bounds.")

            # ── Combined NaN check on critical metadata ──────
            nan_fields = []
            for crit in ('storm_intensity_ms', 'storm_mslp_hpa', 'tc_category', 'center_from_tc_vitals'):
                if crit in f.attrs and is_nan_value(f.attrs[crit]):
                    nan_fields.append(crit)
                    
            if len(nan_fields) >= 2:
                issues.append(
                    f"[GLOBAL] {len(nan_fields)} critical metadata field(s) "
                    f"are NaN simultaneously: {', '.join(nan_fields)} — "
                    f"best-track matching likely failed for this cycle."
                )

            # ── 1b: Expected metadata (must exist and be non-empty) ─────
            for attr in EXPECTED_METADATA:
                if attr not in f.attrs:
                    issues.append(f"[GLOBAL] Missing expected attribute: '{attr}'")
                    continue

                raw_val = f.attrs[attr]
                
                if isinstance(raw_val, (str, bytes, np.bytes_)) and decode_attr(raw_val).strip() == '':
                    issues.append(f"[GLOBAL] Expected attribute '{attr}' is empty.")
                    continue

                if attr in ('time_coverage_start', 'time_coverage_end'):
                    val = decode_attr(raw_val)
                    if not is_nan_value(raw_val):
                        try:
                            datetime.datetime.strptime(val, "%Y-%m-%dT%H:%M:%SZ")
                        except ValueError:
                            issues.append(f"[GLOBAL] '{attr}' is not valid ISO format: '{val}'")

                elif attr in ('geospatial_lat_max', 'geospatial_lat_min',
                              'geospatial_lon_max', 'geospatial_lon_min',
                              'storm_motion_speed_kt', 'storm_motion_heading_deg'):
                    if not is_nan_value(raw_val) and not isinstance(raw_val, (float, np.floating, int, np.integer)):
                        issues.append(f"[GLOBAL] '{attr}' must be a native numeric type. Got {type(raw_val)}")

                # --- V10: Verify the internal metadata link to the JSON sidecar ---
                elif attr == 'Virtual_Manifest':
                    val = decode_attr(raw_val)
                    expected_json = filename.replace('.hdf5', '.json')
                    if val != expected_json:
                        issues.append(
                            f"[GLOBAL] 'Virtual_Manifest' attribute ({val}) "
                            f"does not match actual filename ({expected_json})."
                        )
                # ------------------------------------------------------------------

                elif attr in ('existing_groups', 'expected_groups'):
                    if not is_nan_value(raw_val) and not isinstance(raw_val, (np.ndarray, list)):
                        issues.append(f"[GLOBAL] '{attr}' must be a native string array. Got {type(raw_val)}")


            # ------------------------------------------------------------------
            # CHECK 2 & 3: Root structure & banned instruments
            # ------------------------------------------------------------------
            root_groups = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
            if not root_groups:
                issues.append("[STRUCTURE] File contains no groups — only global attrs.")

            for key in f.keys():
                obj = f[key]
                if not isinstance(obj, h5py.Group):
                    issues.append(f"[STRUCTURE] Root item '{key}' is a Dataset, not a Group.")
                    continue

                if any(b in key.lower() for b in BANNED_INSTRUMENTS):
                    issues.append(f"[BANNED] Banned instrument in group name: '{key}'")

                group = obj
                summary['groups'].append(key)

                # --------------------------------------------------------------
                # CHECK 4: obs_count
                # --------------------------------------------------------------
                if 'obs_count' not in group.attrs:
                    issues.append(f"[{key}] Missing 'obs_count' attribute.")
                    expected_length = -1
                else:
                    try:
                        expected_length = int(group.attrs['obs_count'])
                        if expected_length <= 0:
                            issues.append(f"[{key}] obs_count = {expected_length} — empty group.")
                    except (ValueError, TypeError):
                        issues.append(f"[{key}] obs_count invalid: '{group.attrs['obs_count']}'")
                        expected_length = -1

                summary['obs_counts'][key] = expected_length
                group_vars = set(group.keys())

                # --------------------------------------------------------------
                # CHECK 5-9: Dataset structure, sentinels, missing values, error parity
                # --------------------------------------------------------------
                for item_name in group_vars:
                    dset = group[item_name]

                    if not isinstance(dset, h5py.Dataset):
                        issues.append(f"[{key}] Nested group '{item_name}' found — must be fully flat.")
                        continue

                    if is_track_group(key) and np.issubdtype(dset.dtype, np.number) and dset.size > 0:
                        if np.all(np.isnan(dset[:])):
                            issues.append(f"[{key}/{item_name}] Dataset is entirely NaN in track group.")

                    if is_track_group(key) and item_name.lower() in ('lon', 'clon'):
                        if np.issubdtype(dset.dtype, np.number) and dset.size > 0:
                            data = dset[:]
                            bad = np.nansum(data > 0)
                            if bad > 0:
                                issues.append(f"[{key}/{item_name}] {bad} positive longitude(s).")

                    if expected_length >= 0 and len(dset.shape) > 0:
                        if dset.shape[0] != expected_length:
                            issues.append(f"[{key}/{item_name}] Length mismatch.")

                    if 'fill_value' not in dset.attrs:
                        issues.append(f"[{key}/{item_name}] Missing 'fill_value' attribute.")

                    if np.issubdtype(dset.dtype, np.number) and dset.size > 0:
                        data = dset[:]
                        sentinels_to_check = SHIPS_SENTINEL_VALUES if key == 'ships_params' else SENTINEL_VALUES
                        for sv in sentinels_to_check:
                            hits = int(np.sum(data == sv))
                            if hits > 0:
                                issues.append(f"[{key}/{item_name}] Sentinel {sv} found {hits} time(s).")

                    if item_name.lower() in VALID_BOUNDS and np.issubdtype(dset.dtype, np.number) and dset.size > 0:
                        lo, hi = VALID_BOUNDS[item_name.lower()]
                        data = dset[:]
                        valid = data[~np.isnan(data)]
                        bad = int(np.sum((valid < lo) | (valid > hi)))
                        if bad > 0:
                            issues.append(f"[{key}/{item_name}] {bad} value(s) outside bounds [{lo}, {hi}].")
                    
                    # --- NEW: V10 CF-Time Attribute Check ---
                    if item_name == 'time':
                        expected_units = 'seconds since 1900-01-01 00:00:00Z'
                        actual_units = decode_attr(dset.attrs.get('units', b''))
                        if actual_units != expected_units:
                            issues.append(
                                f"[{key}/{item_name}] Time units missing/incorrect. "
                                f"Expected '{expected_units}', got '{actual_units}'."
                            )

                    if is_err_field(item_name):
                        long_name = decode_attr(dset.attrs.get('long_name', b'')).lower()
                        if 'error estimate' not in long_name:
                            issues.append(f"[{key}/{item_name}] Error field missing 'long_name'.")

                        base_var = item_name[:-3]
                        if base_var in group:
                            base_data = group[base_var][:]
                            err_data  = dset[:]
                            base_nans = np.isnan(base_data)
                            err_nans  = np.isnan(err_data)

                            non_nan_err = err_data[~err_nans]
                            is_constant_err = (len(non_nan_err) == 0 or bool(np.all(non_nan_err == non_nan_err[0])))

                            d1_violations = int((~err_nans[base_nans]).sum())
                            if base_nans.any() and d1_violations > 0:
                                issues.append(f"[{key}/{item_name}] NaN parity violation (base->err).")

                            if not is_constant_err:
                                d2_violations = int((~base_nans[err_nans]).sum())
                                if err_nans.any() and d2_violations > 0:
                                    issues.append(f"[{key}/{item_name}] NaN parity violation (err->base).")

                # --------------------------------------------------------------
                # CHECK 10: Error coverage
                # --------------------------------------------------------------
                if not is_track_group(key) and not is_ships_group(key):
                    for item_name in group_vars:
                        dset = group[item_name]
                        if not isinstance(dset, h5py.Dataset): continue
                        if is_coord(item_name) or is_track_product(item_name) or is_err_field(item_name): continue

                        has_err_array = f"{item_name}err" in group_vars
                        has_err_attr  = 'error_estimate_for_data_assimilation' in dset.attrs

                        if not (has_err_array or has_err_attr):
                            issues.append(f"[{key}/{item_name}] Data field has no error coverage.")

            # ------------------------------------------------------------------
            # CHECK 11: Spline track flight-level pressure
            # ------------------------------------------------------------------
            if SPLINE_TRACK_GROUP in f:
                spline_grp = f[SPLINE_TRACK_GROUP]
                if isinstance(spline_grp, h5py.Group):
                    try:
                        spline_oc = int(spline_grp.attrs.get('obs_count', 0))
                    except (ValueError, TypeError):
                        spline_oc = 0

                    if spline_oc > 0:
                        if 'pres' not in spline_grp:
                            issues.append(f"[{SPLINE_TRACK_GROUP}] Missing 'pres' dataset.")
                        else:
                            pres_dset = spline_grp['pres']
                            pres_data = pres_dset[:]
                            
                            pres_units = decode_attr(pres_dset.attrs.get('units', b''))
                            if pres_units != 'hPa':
                                issues.append(f"[{SPLINE_TRACK_GROUP}/pres] units is '{pres_units}', expected 'hPa'.")

                            pres_ln = decode_attr(pres_dset.attrs.get('long_name', b'')).lower()
                            if 'flight-level' not in pres_ln:
                                issues.append(f"[{SPLINE_TRACK_GROUP}/pres] long_name missing 'flight-level'.")

                            if np.issubdtype(pres_data.dtype, np.number) and pres_data.size > 0:
                                non_nan = pres_data[~np.isnan(pres_data)]
                                if len(non_nan) == 0:
                                    issues.append(f"[{SPLINE_TRACK_GROUP}/pres] All values are NaN.")
                                elif not np.all(non_nan == non_nan[0]):
                                    issues.append(f"[{SPLINE_TRACK_GROUP}/pres] Advisory: pres is not constant.")
                                else:
                                    val = float(non_nan[0])
                                    lo, hi = SPLINE_PRES_BOUNDS
                                    if val < lo or val > hi:
                                        issues.append(f"[{SPLINE_TRACK_GROUP}/pres] Value {val} outside bounds.")

            # ------------------------------------------------------------------
            # CHECK 12 & 13: Entirely NaN data or location columns
            # ------------------------------------------------------------------
            for key in f.keys():
                obj = f[key]
                if not isinstance(obj, h5py.Group) or is_track_group(key) or is_ships_group(key):
                    continue
                
                try: oc = int(obj.attrs.get('obs_count', 0))
                except (ValueError, TypeError): oc = 0
                if oc <= 0: continue

                data_cols = [n for n in obj.keys() if isinstance(obj[n], h5py.Dataset) and not is_coord(n) and not is_err_field(n)]
                if data_cols:
                    all_nan = True
                    for col in data_cols:
                        d = obj[col][:]
                        if np.issubdtype(d.dtype, np.number) and d.size > 0 and not np.all(np.isnan(d)):
                            all_nan = False; break
                    if all_nan:
                        issues.append(f"[{key}] All data variable(s) are entirely NaN.")

                for col_name in obj.keys():
                    dset = obj[col_name]
                    if not isinstance(dset, h5py.Dataset) or col_name.lower() not in LOCATION_VARS: continue
                    if np.issubdtype(dset.dtype, np.number) and dset.size > 0 and np.all(np.isnan(dset[:])):
                        issues.append(f"[{key}/{col_name}] Location column is entirely NaN.")

            # ------------------------------------------------------------------
            # CHECK 14 & 15 & 16: Category, encoding artifacts, dtype consistency
            # ------------------------------------------------------------------
            if 'tc_category' in f.attrs:
                tc_val = decode_attr(f.attrs['tc_category']).strip()
                if tc_val not in VALID_TC_CATEGORIES:
                    issues.append(f"[GLOBAL] tc_category '{tc_val}' unrecognised.")

            for attr_name in f.attrs.keys():
                attr_val = str(f.attrs[attr_name])
                if attr_val.startswith("b'") or attr_val.startswith('b"'):
                    issues.append(f"[GLOBAL] Attribute '{attr_name}' contains byte-string prefix.")

            for key in f.keys():
                obj = f[key]
                if not isinstance(obj, h5py.Group): continue
                for dset_name in obj.keys():
                    dset = obj[dset_name]
                    if isinstance(dset, h5py.Dataset) and np.issubdtype(dset.dtype, np.number) and dset.dtype != np.float64:
                        issues.append(f"[{key}/{dset_name}] Advisory: dtype is {dset.dtype}, expected float64.")

            # ------------------------------------------------------------------
            # CHECK 17: Time monotonicity in track groups
            # ------------------------------------------------------------------
            for key in f.keys():
                obj = f[key]
                if not isinstance(obj, h5py.Group) or not is_track_group(key): continue
                if 'time' in obj and isinstance(obj['time'], h5py.Dataset) and obj['time'].size >= 2:
                    t = obj['time'][:]
                    if np.issubdtype(t.dtype, np.number):
                        t_valid = t[~np.isnan(t)]
                        if len(t_valid) >= 2 and np.sum(np.diff(t_valid) < 0) > 0:
                            issues.append(f"[{key}/time] Non-monotonic timestamps found.")

            # ------------------------------------------------------------------
            # CHECK 18: Storm-center proximity filter
            # ------------------------------------------------------------------
            center_raw = f.attrs.get('center_from_tc_vitals', None)
            clat_val = clon_val = None
            if isinstance(center_raw, (np.ndarray, list)) and len(center_raw) == 2:
                clat_val, clon_val = float(center_raw[0]), float(center_raw[1])

            if clat_val is not None and clon_val is not None:
                for key in f.keys():
                    obj = f[key]
                    if not isinstance(obj, h5py.Group) or is_track_group(key) or is_ships_group(key): continue

                    gcl = {n.lower(): n for n in obj.keys()}
                    lat_col = next((gcl[k] for k in ['lat', 'latitude'] if k in gcl), None)
                    lon_col = next((gcl[k] for k in ['lon', 'longitude'] if k in gcl), None)
                    if not (lat_col and lon_col): continue

                    lat_dset, lon_dset = obj[lat_col], obj[lon_col]
                    if not (isinstance(lat_dset, h5py.Dataset) and isinstance(lon_dset, h5py.Dataset)): continue
                    if lat_dset.size == 0: continue

                    lats, lons = lat_dset[:], lon_dset[:]
                    valid = ~(np.isnan(lats) | np.isnan(lons))
                    if not valid.any(): continue

                    outside = (
                        (lats[valid] < clat_val - OBS_PROXIMITY_DEG) | (lats[valid] > clat_val + OBS_PROXIMITY_DEG) |
                        (lons[valid] < clon_val - OBS_PROXIMITY_DEG) | (lons[valid] > clon_val + OBS_PROXIMITY_DEG)
                    )
                    n_outside = int(outside.sum())
                    if n_outside > 0:
                        issues.append(f"[{key}] {n_outside} observation(s) outside ±{OBS_PROXIMITY_DEG}° proximity.")

            # ------------------------------------------------------------------
            # CHECK 19: ships_params group validation
            # ------------------------------------------------------------------
            if SHIPS_GROUP_NAME in f:
                sp = f[SHIPS_GROUP_NAME]
                if isinstance(sp, h5py.Group):
                    for req_attr in SHIPS_REQUIRED_ATTRS:
                        if req_attr not in sp.attrs:
                            issues.append(f"[{SHIPS_GROUP_NAME}] Missing required attribute: '{req_attr}'")

                    try:
                        sp_oc = int(sp.attrs.get('obs_count', -1))
                        if sp_oc != 1: issues.append(f"[{SHIPS_GROUP_NAME}] obs_count={sp_oc}, expected 1.")
                    except (ValueError, TypeError): pass

                    sp_vars = set(k for k in sp.keys() if isinstance(sp[k], h5py.Dataset))
                    missing_preds = SHIPS_EXPECTED_DATASETS - sp_vars
                    if missing_preds:
                        issues.append(f"[{SHIPS_GROUP_NAME}] {len(missing_preds)} predictor(s) missing.")

                    for col in sp_vars:
                        dset = sp[col]
                        if dset.shape != (1,): issues.append(f"[{SHIPS_GROUP_NAME}/{col}] Shape {dset.shape}, expected (1,).")
                        if np.issubdtype(dset.dtype, np.number) and dset.dtype != np.float64:
                            issues.append(f"[{SHIPS_GROUP_NAME}/{col}] Advisory: dtype is {dset.dtype}, expected float64.")
                        if 'units' not in dset.attrs: issues.append(f"[{SHIPS_GROUP_NAME}/{col}] Missing 'units' attr.")
                        if 'long_name' not in dset.attrs: issues.append(f"[{SHIPS_GROUP_NAME}/{col}] Missing 'long_name' attr.")

    except Exception as e:
        issues.append(f"[FATAL] Could not read file: {e}")

    return issues, summary


# =============================================================================
# RUNNER
# =============================================================================

def run_validation(target_dir, issues_report="validation_issues.csv", counts_report="validation_obs_counts.csv"):
    print(f"🔍 Strict AI-Ready Validation (v10)")
    print(f"   Directory : {target_dir}")
    print(f"   Basin     : {_BASIN_LABEL.get(BASIN_FILTER, BASIN_FILTER)}")
    print(f"   Sentinels : {SENTINEL_VALUES}\n")

    h5_files = sorted(glob.glob(os.path.join(target_dir, "**", "*.hdf5"), recursive=True))
    if not h5_files:
        print("❌ No .hdf5 files found.")
        return

    all_count = len(h5_files)
    h5_files = [f for f in h5_files if should_validate_file(f)]
    n_skipped = all_count - len(h5_files)
    if n_skipped > 0:
        print(f"   Skipped {n_skipped} file(s) outside basin.")
    print(f"   Validating {len(h5_files)} file(s).\n")

    total_files = len(h5_files)
    failed_files = 0
    all_summaries = []
    all_issues = []

    for i, filepath in enumerate(h5_files):
        issues, summary = validate_file(filepath)
        all_summaries.append(summary)

        if issues:
            failed_files += 1
            for issue in issues:
                severity = "Advisory" if "Advisory" in issue else "Error"
                all_issues.append({"Filename": os.path.basename(filepath), "Severity": severity, "Issue": issue})

        if (i + 1) % 500 == 0:
            print(f"   [Progress] {i+1}/{total_files} files validated...")

    # CROSS-FILE CHECK: storm_id naming convention
    from collections import defaultdict
    group_ids = defaultdict(set)
    file_meta = {}

    for fp in h5_files:
        fname = os.path.basename(fp)
        m = re.search(r'HRDOBS_([A-Z0-9]+)[._](\d{4})(\d{2})(\d{2})', fname)
        if not m: continue
        storm_id, year = m.group(1), int(m.group(2))
        nhc_m = re.search(r'(\d+[A-Z]+)$', storm_id)
        if not nhc_m: continue
        nhc_code = nhc_m.group(1)
        group_ids[(year, nhc_code)].add(storm_id)
        file_meta[fname] = (year, nhc_code, storm_id)

    def name_tier(sid):
        mp = re.match(r'([A-Za-z]+)', sid)
        p = mp.group(1).upper() if mp else ''
        if p in _NHC_SPELLED_NUMBERS: return 2
        if p in _NHC_INVEST: return 1
        return 0

    naming_issues = []
    for (year, nhc_code), sids in group_ids.items():
        if len(sids) < 2: continue
        best_tier = min(name_tier(s) for s in sids)
        for fname, (fy, fn, fid) in file_meta.items():
            if fy != year or fn != nhc_code: continue
            if name_tier(fid) > best_tier:
                category = 'INVEST placeholder' if fid.upper().startswith('INVEST') else 'number-word placeholder'
                naming_issues.append({
                    "Filename": fname, "Severity": "Error",
                    "Issue": f"[GLOBAL] storm_id '{fid}' uses a {category} prefix but a higher-tier name exists."
                })

    if naming_issues:
        affected = len({r['Filename'] for r in naming_issues})
        failed_files += affected
        all_issues.extend(naming_issues)

    if all_issues:
        pd.DataFrame(all_issues).to_csv(issues_report, index=False)

    print("\n" + "=" * 60)
    if failed_files == 0:
        print(f"✅ PASSED — all {total_files} files passed all checks.")
    else:
        error_count = sum(1 for r in all_issues if r["Severity"] == "Error")
        advisor_count = sum(1 for r in all_issues if r["Severity"] == "Advisory")
        print(f"⚠️  FAILED — {failed_files}/{total_files} files had issues.")
        print(f"   Hard errors : {error_count}")
        print(f"   Advisories  : {advisor_count}")
        print(f"   Details     → {issues_report}")
    print("=" * 60)

    count_rows = []
    for s in all_summaries:
        for grp, cnt in s['obs_counts'].items():
            count_rows.append({'Filename': s['filename'], 'Group': grp, 'Obs_Count': cnt})

    if count_rows:
        df_counts = pd.DataFrame(count_rows)
        df_counts.to_csv(counts_report, index=False)
        summary_stats = (
            df_counts[df_counts['Obs_Count'] > 0].groupby('Group')['Obs_Count']
            .agg(Files='count', Min='min', Median='median', Max='max', Total='sum')
            .reset_index().sort_values('Total', ascending=False)
        )
        print(f"\n📊 Obs-count report → {counts_report}\n")
        print("Per-group obs summary across all files:")
        print(summary_stats.to_string(index=False))

    ships_present = ships_absent = 0
    pred_valid = {col: 0 for col in SHIPS_EXPECTED_DATASETS}
    pred_total = {col: 0 for col in SHIPS_EXPECTED_DATASETS}

    for fp in h5_files:
        try:
            with h5py.File(fp, 'r') as f:
                if SHIPS_GROUP_NAME not in f or not isinstance(f[SHIPS_GROUP_NAME], h5py.Group):
                    ships_absent += 1; continue
                ships_present += 1
                sp = f[SHIPS_GROUP_NAME]
                for col in SHIPS_EXPECTED_DATASETS:
                    if col in sp and isinstance(sp[col], h5py.Dataset):
                        pred_total[col] += 1
                        data = sp[col][:]
                        if data.size > 0 and not np.all(np.isnan(data)): pred_valid[col] += 1
        except Exception: continue

    if ships_present + ships_absent > 0:
        print(f"\n📊 SHIPS Parameters Summary")
        print(f"   Files with ships_params  : {ships_present} ({100*ships_present/(ships_present+ships_absent):.1f}%)")
        print(f"   Files without            : {ships_absent}")
        if ships_present > 0:
            print(f"\n   Per-predictor coverage (non-NaN):")
            for col in sorted(SHIPS_EXPECTED_DATASETS):
                n_v, n_t = pred_valid.get(col, 0), pred_total.get(col, 0)
                pct = 100.0 * n_v / n_t if n_t > 0 else 0.0
                print(f"   {col:<22}  {n_v:>6,}  {n_t:>6,}  {pct:>5.1f}%")

if __name__ == "__main__":
    TARGET_DIR = "AI_ready_dataset/HRDOBS_hdf5"
    run_validation(TARGET_DIR)
    