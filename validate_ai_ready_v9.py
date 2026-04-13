import os
import glob
import re
import datetime
import h5py
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION — must stay in sync with make_ai_ready_batch_v9.py
# =============================================================================

CRITICAL_METADATA = [
    'storm_name', 'storm_id', 'storm_datetime', 'storm_epoch',
    'center_from_tc_vitals', 'tc_category',
    'storm_intensity_ms', 'storm_mslp_hpa', 'radius_of_maximum_wind_km',
]

EXPECTED_METADATA = [
    'creator_email', 'creator_name',
    'geospatial_lat_max', 'geospatial_lat_min',
    'geospatial_lat_units',
    'geospatial_lon_max', 'geospatial_lon_min',
    'geospatial_lon_units',
    'platforms', 'storm_motion',
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

# Physical bounds for QC — must stay in sync with VALID_BOUNDS in
# make_ai_ready_batch_v8.py.  Any value outside these ranges should have been
# replaced with NaN during conversion; finding one here is a bug.
#
# v8 additions:
#   'latitude' / 'longitude' — long-form aliases added so CHECK 8b fires
#     regardless of how a group names its coordinate columns.
#   'height' / 'altitude' / 'ght' — vertical coordinate bounds added to
#     catch sub-surface (< 0 m) and absurdly high (> 20 000 m) values.
#   Note: 'elev' is intentionally excluded — it is a TDR beam elevation angle
#   in degrees, not an altitude.
VALID_BOUNDS = {
    'lat':       (-90.0,    90.0),
    'latitude':  (-90.0,    90.0),
    'lon':       (-180.0,  180.0),
    'longitude': (-180.0,  180.0),
    'time':      (19900000000000.0, 20300000000000.0),   # YYYYMMDDHHMMSS
    'sfcp':      (80000.0, 110000.0),                    # Pa
    'height':    (0.0, 20000.0),                         # m
    'altitude':  (0.0, 20000.0),                         # m
    'ght':       (0.0, 20000.0),                         # m
}

# v8: Maximum allowed lat/lon box half-width (degrees) from storm center for
# observation data.  Must match OBS_PROXIMITY_DEG in make_ai_ready_batch_v8.py.
# Track groups are exempt from this check.
OBS_PROXIMITY_DEG = 10.0

# v8: Location-type columns whose presence as an entirely-NaN array in an
# output group indicates a QC failure.  Any group with an all-NaN column
# from this set should have been dropped during conversion.
# 'elev' (TDR beam angle) and 'p' (used as both coord and obs variable)
# are intentionally excluded.
LOCATION_VARS = {'lat', 'latitude', 'lon', 'longitude', 'height', 'altitude', 'ght'}

# =============================================================================
# BASIN FILTERING — must stay in sync with make_ai_ready_batch_v8.py
# =============================================================================
BASIN_FILTER = 'ATL'

_BASIN_SUFFIX_MAP = {
    'L': 'ATL',
    'E': 'EPAC',
}
_BASIN_LABEL = {'ATL': 'Atlantic', 'EPAC': 'East Pacific', 'ALL': 'All Basins'}

# =============================================================================
# SPLINE TRACK PRESSURE VALIDATION
# =============================================================================
# Reasonable range for flight-level pressure in hPa.
# Aircraft fly between ~400 hPa (high-altitude jet, G-IV excluded) and
# ~1013 hPa (near surface for very weak systems).
SPLINE_PRES_BOUNDS = (400.0, 1100.0)

# Valid tc_category tokens — any value not in this set is a conversion bug.
VALID_TC_CATEGORIES = {
    'TD', 'TS', 'H1', 'H2', 'H3', 'H4', 'H5',
    'SS', 'SD', 'EX', 'LO', 'DB', 'WV', 'HU', 'NaN',
}

# NHC placeholder prefixes — after renaming, no AI-ready file's storm_id
# global attribute should still carry one of these prefixes.
_NHC_SPELLED_NUMBERS = {
    'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE',
    'TEN', 'ELEVEN', 'TWELVE', 'THIRTEEN', 'FOURTEEN', 'FIFTEEN', 'SIXTEEN',
    'SEVENTEEN', 'EIGHTEEN', 'NINETEEN', 'TWENTY', 'TWENTYONE', 'TWENTYTWO',
    'TWENTYTHREE', 'TWENTYFOUR', 'TWENTYFIVE',
}
_NHC_INVEST       = {'INVEST'}
_NHC_NUMBER_WORDS = _NHC_SPELLED_NUMBERS | _NHC_INVEST

# ── Field classification ──────────────────────────────────────────────────────
#
# COORD_VARS: positional / geometric descriptors that anchor observations.
#   These carry no assimilation error and are never expected to have a
#   corresponding *err field.
#
# TRACK_PRODUCT_VARS: derived products in track groups (best track, spline,
#   vortex message). Not direct assimilation observations — error fields
#   are not required.
#
# Everything else is treated as a DATA field and MUST have error coverage —
# either a native *err array in the group, OR an
# error_estimate_for_data_assimilation attribute on the dataset itself.

COORD_VARS = {
    'time', 'lat', 'latitude', 'lon', 'longitude',
    'ght',          # geopotential height  (vertical coordinate)
    'p',            # pressure             (vertical coordinate for dropsondes/flight-level)
    'az',           # TDR azimuth angle    (beam geometry)
    'elev',         # TDR elevation angle  (beam geometry)
    'clat', 'clon', # track center position
    'rr',           # SFMR rain rate (used internally to derive spd error, not an assimilation obs)
}

TRACK_PRODUCT_VARS = {'vmax', 'pmin', 'rmw', 'sfcp', 'sfcspd', 'sfcdir', 'pres'}

# Track group name prefix — used to skip error-coverage checks on track groups
TRACK_GROUP_PREFIX = 'track_'

# Spline track group name — used for pres validation
SPLINE_TRACK_GROUP = 'track_spline_track'

# =============================================================================
# SHIPS PARAMETERS GROUP (v9)
# =============================================================================
# Group name written by make_ai_ready_batch_v9.py when a matching SHIPS cycle
# is found.  This group has different validation rules from observation groups:
# it contains exactly 1 value per predictor, has no error fields, and has no
# location columns — so several standard checks are intentionally skipped.
SHIPS_GROUP_NAME = 'ships_params'

# Required group-level attributes on ships_params
SHIPS_REQUIRED_ATTRS = {'obs_count', 'source', 'ships_atcf_id', 'ships_datetime_utc'}

# All 25 SHIPS predictor datasets expected inside ships_params.
# Must stay in sync with SHIPS_PREDICTOR_META in make_ai_ready_batch_v9.py.
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
    """Decode a scalar or 1-element array HDF5 attribute to a plain string.

    Multi-element string arrays are joined with ', ' so no elements are
    silently dropped (mirrors decode_attr in make_ai_ready_batch_v8.py).
    """
    if isinstance(val, (np.ndarray, list)):
        if len(val) == 0:
            return ''
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


def is_track_group(group_name):
    return group_name.lower().startswith(TRACK_GROUP_PREFIX)


def is_ships_group(group_name):
    # Return True if the group is the ships_params parameter group (v9).
    return group_name.lower() == SHIPS_GROUP_NAME


def is_coord(var_name):
    return var_name.lower() in COORD_VARS


def is_track_product(var_name):
    return var_name.lower() in TRACK_PRODUCT_VARS


def is_err_field(var_name):
    return var_name.lower().endswith('err')


def detect_basin(storm_id):
    """Detect basin from storm_id (e.g. 'BERYL02L' → 'ATL')."""
    m = re.search(r'\d+([A-Z]+)$', storm_id.upper())
    if m:
        return _BASIN_SUFFIX_MAP.get(m.group(1), 'UNKNOWN')
    return 'UNKNOWN'


def should_validate_file(filepath):
    """Check whether a file should be validated under the current BASIN_FILTER."""
    fname = os.path.basename(filepath)
    m = re.search(r'HRDOBS_([A-Z0-9]+)[._]', fname)
    if not m:
        return True   # can't parse — validate anyway
    storm_id = m.group(1)
    basin = detect_basin(storm_id)
    if BASIN_FILTER == 'ALL':
        return True
    return basin == BASIN_FILTER


# =============================================================================
# PER-FILE VALIDATION
# =============================================================================

def validate_file(filepath):
    """
    Run all structural and data-quality checks on one AI-ready HDF5 file.

    Returns:
        issues  — list of issue strings (empty = passed)
        summary — dict for the cross-file count report
    """
    issues  = []
    summary = {
        'filename':   os.path.basename(filepath),
        'groups':     [],
        'obs_counts': {},
    }

    try:
        with h5py.File(filepath, 'r') as f:

            # ------------------------------------------------------------------
            # CHECK 1: All 23 global attributes — present, non-empty, and
            # correctly typed/formatted.
            #
            # CRITICAL fields must not be 'NaN' — the conversion should
            # have skipped files with incomplete critical metadata.
            #
            # EXPECTED fields must be present and non-empty — missing ones
            # are flagged as advisories.
            # ------------------------------------------------------------------

            # ── 1a: Critical metadata (must exist, non-empty, non-NaN) ──
            for attr in CRITICAL_METADATA:
                if attr not in f.attrs:
                    issues.append(f"[GLOBAL] Missing critical attribute: '{attr}'")
                    continue

                val = decode_attr(f.attrs[attr])

                if val.strip() == '':
                    issues.append(
                        f"[GLOBAL] Critical attribute '{attr}' is empty."
                    )
                    continue

                if val.strip() in ('NaN', 'nan'):
                    issues.append(
                        f"[GLOBAL] Critical attribute '{attr}' is 'NaN' "
                        f"— file should have been excluded during conversion."
                    )
                    continue

                # Per-attribute format validation
                if attr == 'storm_name':
                    if not re.match(r'^[A-Z]+$', val):
                        issues.append(
                            f"[GLOBAL] 'storm_name' is not a valid "
                            f"uppercase alpha name: '{val}'"
                        )

                elif attr == 'storm_id':
                    if not re.match(r'^[A-Z]+\d+[A-Z]+$', val):
                        issues.append(
                            f"[GLOBAL] 'storm_id' does not match expected "
                            f"pattern (NAME + NHC_CODE): '{val}'"
                        )

                elif attr == 'storm_datetime':
                    try:
                        datetime.datetime.strptime(val, "%Y-%m-%dT%H:%M:%SZ")
                    except ValueError:
                        issues.append(
                            f"[GLOBAL] 'storm_datetime' is not valid "
                            f"ISO format (YYYY-MM-DDTHH:MM:SSZ): '{val}'"
                        )

                elif attr == 'storm_epoch':
                    try:
                        epoch = int(val)
                        if epoch < 631152000 or epoch > 2051222400:
                            issues.append(
                                f"[GLOBAL] 'storm_epoch' value {epoch} "
                                f"outside reasonable range (1990–2035)."
                            )
                    except (ValueError, TypeError):
                        issues.append(
                            f"[GLOBAL] 'storm_epoch' is not a valid "
                            f"integer: '{val}'"
                        )

                elif attr == 'tc_category':
                    pass  # token validated in CHECK 14

                elif attr in ('storm_intensity_ms', 'storm_mslp_hpa',
                              'radius_of_maximum_wind_km'):
                    try:
                        fv = float(val)
                        if attr == 'storm_intensity_ms' and (fv < 0 or fv > 100):
                            issues.append(
                                f"[GLOBAL] '{attr}' value {fv} m/s "
                                f"outside reasonable range [0, 100]."
                            )
                        elif attr == 'storm_mslp_hpa' and (fv < 850 or fv > 1050):
                            issues.append(
                                f"[GLOBAL] '{attr}' value {fv} hPa "
                                f"outside reasonable range [850, 1050]."
                            )
                        elif attr == 'radius_of_maximum_wind_km' and (fv < 0 or fv > 750):
                            issues.append(
                                f"[GLOBAL] '{attr}' value {fv} km "
                                f"outside reasonable range [0, 750]."
                            )
                    except ValueError:
                        issues.append(
                            f"[GLOBAL] '{attr}' is not a valid float: '{val}'"
                        )

                elif attr == 'center_from_tc_vitals':
                    parts = val.split(',')
                    if len(parts) != 2:
                        issues.append(
                            f"[GLOBAL] '{attr}' format invalid "
                            f"(expected 'lat, lon'): '{val}'"
                        )
                    else:
                        try:
                            clat = float(parts[0].strip())
                            clon = float(parts[1].strip())
                            if clat < -90 or clat > 90:
                                issues.append(
                                    f"[GLOBAL] '{attr}' latitude "
                                    f"{clat} outside [-90, 90]."
                                )
                            if clon < -180 or clon > 180:
                                issues.append(
                                    f"[GLOBAL] '{attr}' longitude "
                                    f"{clon} outside [-180, 180]."
                                )
                        except ValueError:
                            issues.append(
                                f"[GLOBAL] '{attr}' contains "
                                f"non-float coordinates: '{val}'"
                            )

            # ── 1a-post: Combined NaN check on critical metadata ──────
            # If multiple critical fields are simultaneously NaN, the
            # best-track matching failed entirely — this file should not
            # have been published.
            nan_fields = []
            for crit in ('storm_intensity_ms', 'storm_mslp_hpa',
                         'tc_category', 'center_from_tc_vitals'):
                if crit in f.attrs:
                    cv = decode_attr(f.attrs[crit]).strip()
                    if cv in ('NaN', 'nan', ''):
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
                    issues.append(
                        f"[GLOBAL] Missing expected attribute: '{attr}'"
                    )
                    continue

                val = decode_attr(f.attrs[attr])
                if val.strip() == '':
                    issues.append(
                        f"[GLOBAL] Expected attribute '{attr}' is empty."
                    )
                    continue

                # Format checks for specific expected fields
                if attr in ('time_coverage_start', 'time_coverage_end'):
                    try:
                        datetime.datetime.strptime(val,
                                                   "%Y-%m-%dT%H:%M:%SZ")
                    except ValueError:
                        issues.append(
                            f"[GLOBAL] '{attr}' is not valid "
                            f"ISO format: '{val}'"
                        )

                elif attr in ('geospatial_lat_max', 'geospatial_lat_min',
                              'geospatial_lon_max', 'geospatial_lon_min'):
                    try:
                        float(val)
                    except ValueError:
                        issues.append(
                            f"[GLOBAL] '{attr}' is not a "
                            f"valid number: '{val}'"
                        )

            # ------------------------------------------------------------------
            # CHECK 2a: File must contain at least one group
            # ------------------------------------------------------------------
            root_groups = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
            if not root_groups:
                issues.append(
                    "[STRUCTURE] File contains no groups — only global "
                    "attributes.  Empty files should not be published."
                )

            # ------------------------------------------------------------------
            # CHECK 2b: All root-level items must be Groups (flat structure)
            # ------------------------------------------------------------------
            for key in f.keys():
                obj = f[key]

                if not isinstance(obj, h5py.Group):
                    issues.append(
                        f"[STRUCTURE] Root item '{key}' is a Dataset, "
                        f"not a Group — flat structure violated."
                    )
                    continue

                # --------------------------------------------------------------
                # CHECK 3: No banned instrument names in group keys
                # --------------------------------------------------------------
                if any(b in key.lower() for b in BANNED_INSTRUMENTS):
                    issues.append(
                        f"[BANNED] Banned instrument in group name: '{key}'"
                    )

                group = obj
                summary['groups'].append(key)

                # --------------------------------------------------------------
                # CHECK 4: obs_count — present, integer, > 0
                # --------------------------------------------------------------
                if 'obs_count' not in group.attrs:
                    issues.append(f"[{key}] Missing 'obs_count' attribute.")
                    expected_length = -1
                else:
                    try:
                        expected_length = int(group.attrs['obs_count'])
                        if expected_length <= 0:
                            issues.append(
                                f"[{key}] obs_count = {expected_length} — "
                                f"empty groups should have been deleted."
                            )
                    except (ValueError, TypeError):
                        issues.append(
                            f"[{key}] obs_count is not a valid integer: "
                            f"'{group.attrs['obs_count']}'"
                        )
                        expected_length = -1

                summary['obs_counts'][key] = expected_length

                # Collect all variable names in this group for use in checks 5–10
                group_vars = set(group.keys())

                # --------------------------------------------------------------
                # CHECK 5: All children must be Datasets (no nested groups)
                # --------------------------------------------------------------
                for item_name in group_vars:
                    dset = group[item_name]

                    if not isinstance(dset, h5py.Dataset):
                        issues.append(
                            f"[{key}] Nested group '{item_name}' found — "
                            f"must be fully flat."
                        )
                        continue

                    # ----------------------------------------------------------
                    # CHECK 5b: Track group datasets must not be entirely NaN
                    # (e.g. sfcdir was never populated in vortex_message)
                    # ----------------------------------------------------------
                    if is_track_group(key) and np.issubdtype(dset.dtype, np.number) and dset.size > 0:
                        if np.all(np.isnan(dset[:])):
                            issues.append(
                                f"[{key}/{item_name}] Dataset is entirely NaN — "
                                f"unpopulated fields should be omitted from track groups."
                            )

                    # ----------------------------------------------------------
                    # CHECK 5c: Longitude sign in track groups
                    # All lon/clon values must be negative (Western Hemisphere).
                    # Positive values indicate the sign was not corrected.
                    # ----------------------------------------------------------
                    if is_track_group(key) and item_name.lower() in ('lon', 'clon'):
                        if np.issubdtype(dset.dtype, np.number) and dset.size > 0:
                            data = dset[:]
                            bad = np.nansum(data > 0)
                            if bad > 0:
                                issues.append(
                                    f"[{key}/{item_name}] {bad} longitude value(s) > 0 — "
                                    f"positive longitudes indicate sign was not corrected."
                                )

                    # ----------------------------------------------------------
                    # CHECK 6: Array length matches obs_count
                    # ----------------------------------------------------------
                    if expected_length >= 0 and len(dset.shape) > 0:
                        if dset.shape[0] != expected_length:
                            issues.append(
                                f"[{key}/{item_name}] Length mismatch: "
                                f"obs_count={expected_length}, "
                                f"array length={dset.shape[0]}."
                            )

                    # ----------------------------------------------------------
                    # CHECK 7: fill_value attribute present on every dataset
                    # ----------------------------------------------------------
                    if 'fill_value' not in dset.attrs:
                        issues.append(
                            f"[{key}/{item_name}] Missing 'fill_value' attribute."
                        )

                    # ----------------------------------------------------------
                    # CHECK 8: Sentinel values must not remain in the data
                    # ----------------------------------------------------------
                    if np.issubdtype(dset.dtype, np.number) and dset.size > 0:
                        data = dset[:]
                        
                        # Apply the specific sentinel list based on the current group
                        sentinels_to_check = SHIPS_SENTINEL_VALUES if key == 'ships_params' else SENTINEL_VALUES
                        
                        for sv in sentinels_to_check:
                            hits = int(np.sum(data == sv))
                            if hits > 0:
                                issues.append(
                                    f"[{key}/{item_name}] Sentinel {sv} found "
                                    f"{hits} time(s) — missing-value replacement "
                                    f"may have failed."
                                )

                    # ----------------------------------------------------------
                    # CHECK 8b: Physical bounds (lat, lon, time, sfcp)
                    # Values outside these ranges should have been replaced with
                    # NaN during conversion; any survivors are a bug.
                    # ----------------------------------------------------------
                    if item_name.lower() in VALID_BOUNDS and \
                            np.issubdtype(dset.dtype, np.number) and dset.size > 0:
                        lo, hi = VALID_BOUNDS[item_name.lower()]
                        data   = dset[:]
                        valid  = data[~np.isnan(data)]
                        bad    = int(np.sum((valid < lo) | (valid > hi)))
                        if bad > 0:
                            issues.append(
                                f"[{key}/{item_name}] {bad} value(s) outside "
                                f"physical bounds [{lo}, {hi}] — bound-violation "
                                f"replacement may have failed."
                            )

                    # ----------------------------------------------------------
                    # CHECK 9: Error field integrity (only for *err fields)
                    # ----------------------------------------------------------
                    if is_err_field(item_name):
                        # 9a: long_name must contain 'error estimate'
                        long_name = decode_attr(
                            dset.attrs.get('long_name', b'')
                        ).lower()
                        if 'error estimate' not in long_name:
                            issues.append(
                                f"[{key}/{item_name}] Error field missing "
                                f"standardized 'long_name' (got: '{long_name}')."
                            )

                        # 9b: Bidirectional NaN parity check.
                        # item_name[:-3] strips 'err': qerr->q, rvelerr->rvel, etc.
                        #
                        # For CONSTANT error fields (attribute-based or hardcoded,
                        # e.g. rvelerr=2.0): only base NaN -> err NaN applies.
                        # A constant error can never be NaN unless we set it, so
                        # the reverse direction is not meaningful to check.
                        #
                        # For NATIVE *err arrays (independently written values,
                        # e.g. qerr, spderr): both directions apply — either being
                        # NaN while the other is valid is a data integrity failure.
                        base_var = item_name[:-3]
                        if base_var in group:
                            base_data = group[base_var][:]
                            err_data  = dset[:]
                            base_nans = np.isnan(base_data)
                            err_nans  = np.isnan(err_data)

                            # Determine if this is a constant or native error field.
                            # A constant error has no NaN values except where base
                            # is NaN — if all err values are either the same constant
                            # or NaN, treat it as constant.
                            non_nan_err = err_data[~err_nans]
                            is_constant_err = (
                                len(non_nan_err) == 0 or
                                bool(np.all(non_nan_err == non_nan_err[0]))
                            )

                            # Direction 1 (all types): base NaN -> err must be NaN
                            d1_violations = int((~err_nans[base_nans]).sum())
                            if base_nans.any() and d1_violations > 0:
                                issues.append(
                                    f"[{key}/{item_name}] NaN parity violation "
                                    f"(base->err): {d1_violations} position(s) "
                                    f"where '{base_var}' is NaN but "
                                    f"'{item_name}' is not."
                                )

                            # Direction 2 (native arrays only): err NaN -> base must be NaN
                            if not is_constant_err:
                                d2_violations = int((~base_nans[err_nans]).sum())
                                if err_nans.any() and d2_violations > 0:
                                    issues.append(
                                        f"[{key}/{item_name}] NaN parity violation "
                                        f"(err->base): {d2_violations} position(s) "
                                        f"where '{item_name}' is NaN but "
                                        f"'{base_var}' is valid."
                                    )

                # --------------------------------------------------------------
                # CHECK 10: Error coverage for every data field
                #
                # Skipped for track groups entirely.
                # For observation groups, every variable that is NOT a coord,
                # NOT a track product, and NOT an *err field itself must have
                # error coverage via one of:
                #   a) a native *err array present in the group
                #   b) an error_estimate_for_data_assimilation attr on the dataset
                # --------------------------------------------------------------
                if not is_track_group(key) and not is_ships_group(key):
                    for item_name in group_vars:
                        dset = group[item_name]
                        if not isinstance(dset, h5py.Dataset):
                            continue
                        if (is_coord(item_name) or
                                is_track_product(item_name) or
                                is_err_field(item_name)):
                            continue

                        has_err_array = f"{item_name}err" in group_vars
                        has_err_attr  = (
                            'error_estimate_for_data_assimilation' in dset.attrs
                        )

                        if not (has_err_array or has_err_attr):
                            issues.append(
                                f"[{key}/{item_name}] Data field has no error "
                                f"coverage: no '{item_name}err' array and no "
                                f"'error_estimate_for_data_assimilation' attribute."
                            )

            # ------------------------------------------------------------------
            # CHECK 11: Spline track flight-level pressure ('pres')
            #
            # If track_spline_track exists and has data, a 'pres' dataset
            # must be present with:
            #   a) units = 'hPa'
            #   b) long_name containing 'flight-level'
            #   c) constant value (all non-NaN values equal)
            #   d) value within reasonable flight-level range
            #   e) array length matching obs_count (covered by CHECK 6)
            #   f) fill_value attribute (covered by CHECK 7)
            # ------------------------------------------------------------------
            if SPLINE_TRACK_GROUP in f:
                spline_grp = f[SPLINE_TRACK_GROUP]
                if isinstance(spline_grp, h5py.Group):
                    spline_oc = spline_grp.attrs.get('obs_count', 0)
                    try:
                        spline_oc = int(spline_oc)
                    except (ValueError, TypeError):
                        spline_oc = 0

                    if spline_oc > 0:
                        # 11a: pres dataset must exist
                        if 'pres' not in spline_grp:
                            issues.append(
                                f"[{SPLINE_TRACK_GROUP}] Missing 'pres' dataset "
                                f"— flight-level pressure should have been "
                                f"injected during conversion (Mode 6)."
                            )
                        else:
                            pres_dset = spline_grp['pres']
                            pres_data = pres_dset[:]

                            # 11b: units attribute must be 'hPa'
                            pres_units = decode_attr(
                                pres_dset.attrs.get('units', b'')
                            )
                            if pres_units != 'hPa':
                                issues.append(
                                    f"[{SPLINE_TRACK_GROUP}/pres] units attribute "
                                    f"is '{pres_units}', expected 'hPa'."
                                )

                            # 11c: long_name must contain 'flight-level'
                            pres_ln = decode_attr(
                                pres_dset.attrs.get('long_name', b'')
                            ).lower()
                            if 'flight-level' not in pres_ln:
                                issues.append(
                                    f"[{SPLINE_TRACK_GROUP}/pres] long_name "
                                    f"does not contain 'flight-level' "
                                    f"(got: '{pres_ln}')."
                                )

                            # 11d: values must be constant
                            if np.issubdtype(pres_data.dtype, np.number) \
                                    and pres_data.size > 0:
                                non_nan = pres_data[~np.isnan(pres_data)]
                                if len(non_nan) == 0:
                                    issues.append(
                                        f"[{SPLINE_TRACK_GROUP}/pres] All values "
                                        f"are NaN — no altitude was assigned."
                                    )
                                elif not np.all(non_nan == non_nan[0]):
                                    unique_vals = np.unique(non_nan)
                                    issues.append(
                                        f"[{SPLINE_TRACK_GROUP}/pres] Advisory: "
                                        f"pres is not constant — "
                                        f"{len(unique_vals)} distinct value(s) "
                                        f"found (expected a single flight level)."
                                    )
                                else:
                                    # 11e: value in reasonable range
                                    val = float(non_nan[0])
                                    lo, hi = SPLINE_PRES_BOUNDS
                                    if val < lo or val > hi:
                                        issues.append(
                                            f"[{SPLINE_TRACK_GROUP}/pres] "
                                            f"Value {val:.1f} hPa outside "
                                            f"expected flight-level range "
                                            f"[{lo:.0f}, {hi:.0f}]."
                                        )

            # ------------------------------------------------------------------
            # CHECK 12: Observation groups where ALL data columns are NaN
            #
            # If obs_count > 0 but every non-coordinate, non-error variable
            # is entirely NaN, the group is useless and should not have been
            # written.  (The conversion QC deletes rows where all obs are NaN,
            # but this catches any that slipped through.)
            # ------------------------------------------------------------------
            for key in f.keys():
                obj = f[key]
                if not isinstance(obj, h5py.Group):
                    continue
                if is_track_group(key) or is_ships_group(key):
                    continue
                oc = obj.attrs.get('obs_count', 0)
                try:
                    oc = int(oc)
                except (ValueError, TypeError):
                    oc = 0
                if oc <= 0:
                    continue

                data_cols = [
                    name for name in obj.keys()
                    if isinstance(obj[name], h5py.Dataset)
                    and not is_coord(name)
                    and not is_err_field(name)
                ]
                if data_cols:
                    all_nan = True
                    for col in data_cols:
                        d = obj[col][:]
                        if np.issubdtype(d.dtype, np.number) and d.size > 0:
                            if not np.all(np.isnan(d)):
                                all_nan = False
                                break
                        else:
                            all_nan = False
                            break
                    if all_nan:
                        issues.append(
                            f"[{key}] All {len(data_cols)} data variable(s) "
                            f"are entirely NaN (obs_count={oc}) — group "
                            f"should have been deleted during conversion."
                        )

            # ------------------------------------------------------------------
            # CHECK 13: All-NaN location columns in observation groups
            #
            # v8: If a group contains a column from LOCATION_VARS (lat, lon,
            # height, altitude, ght, etc.) and that entire column is NaN, the
            # group carries no usable spatial context and should not have been
            # written.  This catches cases where vertical coordinate values
            # were all bad and replaced with NaN by bounds QC, but the rows
            # themselves survived (because lat/lon/time were still valid).
            #
            # Track groups are exempt — they may legitimately omit vertical
            # coordinates.
            # ------------------------------------------------------------------
            for key in f.keys():
                obj = f[key]
                if not isinstance(obj, h5py.Group):
                    continue
                if is_track_group(key) or is_ships_group(key):
                    continue
                oc = obj.attrs.get('obs_count', 0)
                try:
                    oc = int(oc)
                except (ValueError, TypeError):
                    oc = 0
                if oc <= 0:
                    continue

                for col_name in obj.keys():
                    dset = obj[col_name]
                    if not isinstance(dset, h5py.Dataset):
                        continue
                    if col_name.lower() not in LOCATION_VARS:
                        continue
                    if not (np.issubdtype(dset.dtype, np.number) and dset.size > 0):
                        continue
                    if np.all(np.isnan(dset[:])):
                        issues.append(
                            f"[{key}/{col_name}] Location column is entirely NaN "
                            f"(obs_count={oc}) — group should have been deleted "
                            f"during conversion."
                        )

            # ------------------------------------------------------------------
            # CHECK 14: tc_category value validity
            #
            # The tc_category global attribute must be one of the recognised
            # Saffir-Simpson / NHC tokens.  Anything else indicates a
            # resolution failure in the conversion.
            # ------------------------------------------------------------------
            if 'tc_category' in f.attrs:
                tc_val = decode_attr(f.attrs['tc_category']).strip()
                if tc_val not in VALID_TC_CATEGORIES:
                    issues.append(
                        f"[GLOBAL] tc_category '{tc_val}' is not a "
                        f"recognised category token. Expected one of: "
                        f"{', '.join(sorted(VALID_TC_CATEGORIES))}"
                    )

            # ------------------------------------------------------------------
            # CHECK 15: Global attribute encoding artifacts
            #
            # The conversion decodes byte-strings to UTF-8.  If the literal
            # "b'" prefix appears in any attribute value, the decode failed
            # and raw Python repr leaked into the output.
            # ------------------------------------------------------------------
            for attr_name in f.attrs.keys():
                attr_val = str(f.attrs[attr_name])
                if attr_val.startswith("b'") or attr_val.startswith('b"'):
                    issues.append(
                        f"[GLOBAL] Attribute '{attr_name}' contains raw "
                        f"byte-string prefix — encoding decode failed: "
                        f"'{attr_val[:60]}'"
                    )

            # ------------------------------------------------------------------
            # CHECK 16: Dataset dtype consistency
            #
            # The conversion casts all datasets to float64.  Any other
            # numeric type (int32, float32, etc.) indicates a conversion bug.
            # String datasets are allowed in principle but not expected.
            # ------------------------------------------------------------------
            for key in f.keys():
                obj = f[key]
                if not isinstance(obj, h5py.Group):
                    continue
                for dset_name in obj.keys():
                    dset = obj[dset_name]
                    if not isinstance(dset, h5py.Dataset):
                        continue
                    if np.issubdtype(dset.dtype, np.number):
                        if dset.dtype != np.float64:
                            issues.append(
                                f"[{key}/{dset_name}] Advisory: dtype is "
                                f"{dset.dtype}, expected float64."
                            )

            # ------------------------------------------------------------------
            # CHECK 17: Time monotonicity in track groups
            #
            # Track groups (best track, spline track, vortex message) should
            # have monotonically non-decreasing time values.  Non-monotonic
            # timestamps break temporal interpolation downstream.
            # ------------------------------------------------------------------
            for key in f.keys():
                obj = f[key]
                if not isinstance(obj, h5py.Group):
                    continue
                if not is_track_group(key):
                    continue
                if 'time' not in obj:
                    continue
                time_dset = obj['time']
                if not isinstance(time_dset, h5py.Dataset):
                    continue
                if time_dset.size < 2:
                    continue
                t = time_dset[:]
                if np.issubdtype(t.dtype, np.number):
                    # Ignore NaN values for monotonicity check
                    t_valid = t[~np.isnan(t)]
                    if len(t_valid) >= 2:
                        diffs = np.diff(t_valid)
                        n_decreasing = int(np.sum(diffs < 0))
                        if n_decreasing > 0:
                            issues.append(
                                f"[{key}/time] {n_decreasing} non-monotonic "
                                f"step(s) — track timestamps should be "
                                f"monotonically non-decreasing."
                            )

            # ------------------------------------------------------------------
            # CHECK 18: Storm-center proximity filter (v8)
            #
            # Every observation in a non-track group must fall within
            # OBS_PROXIMITY_DEG degrees of the storm center at cycle time.
            # Any survivors indicate the proximity filter in process_obs_leaf
            # did not fire — either a code bug or the center was unavailable
            # during conversion (in which case this check is skipped too).
            # ------------------------------------------------------------------
            center_str = decode_attr(f.attrs.get('center_from_tc_vitals', 'NaN')).strip()
            clat_val = clon_val = None
            if center_str not in ('NaN', 'nan', ''):
                parts = center_str.split(',')
                if len(parts) == 2:
                    try:
                        clat_val = float(parts[0].strip())
                        clon_val = float(parts[1].strip())
                    except ValueError:
                        pass

            if clat_val is not None and clon_val is not None:
                for key in f.keys():
                    obj = f[key]
                    if not isinstance(obj, h5py.Group):
                        continue
                    if is_track_group(key) or is_ships_group(key):
                        continue

                    gcl = {n.lower(): n for n in obj.keys()}
                    lat_col = next((gcl[k] for k in ['lat', 'latitude'] if k in gcl), None)
                    lon_col = next((gcl[k] for k in ['lon', 'longitude'] if k in gcl), None)
                    if not (lat_col and lon_col):
                        continue

                    lat_dset = obj[lat_col]
                    lon_dset = obj[lon_col]
                    if not (isinstance(lat_dset, h5py.Dataset) and
                            isinstance(lon_dset, h5py.Dataset)):
                        continue
                    if lat_dset.size == 0:
                        continue

                    lats = lat_dset[:]
                    lons = lon_dset[:]
                    valid = ~(np.isnan(lats) | np.isnan(lons))
                    if not valid.any():
                        continue

                    outside = (
                        (lats[valid] < clat_val - OBS_PROXIMITY_DEG) |
                        (lats[valid] > clat_val + OBS_PROXIMITY_DEG) |
                        (lons[valid] < clon_val - OBS_PROXIMITY_DEG) |
                        (lons[valid] > clon_val + OBS_PROXIMITY_DEG)
                    )
                    n_outside = int(outside.sum())
                    if n_outside > 0:
                        issues.append(
                            f"[{key}] {n_outside} observation(s) fall outside "
                            f"±{OBS_PROXIMITY_DEG}° of storm center "
                            f"({clat_val:.2f}N, {clon_val:.2f}E) — "
                            f"proximity filter may not have been applied."
                        )

            # ------------------------------------------------------------------
            # CHECK 19: ships_params group validation (v9)
#
            # When the ships_params group is present, verify:
#   a) Required group-level attributes exist
#   b) obs_count == 1 (exactly one cycle-level parameter set)
#   c) All 25 expected predictor datasets are present
#   d) Each dataset has shape (1,), dtype float64, and units/long_name attrs
            # ------------------------------------------------------------------
            if SHIPS_GROUP_NAME in f:
                sp = f[SHIPS_GROUP_NAME]
                if isinstance(sp, h5py.Group):

                    # 19a: Required group attributes
                    for req_attr in SHIPS_REQUIRED_ATTRS:
                        if req_attr not in sp.attrs:
                            issues.append(
                                f"[{SHIPS_GROUP_NAME}] Missing required "
                                f"attribute: '{req_attr}'"
                            )

                    # 19b: obs_count must be exactly 1
                    try:
                        sp_oc = int(sp.attrs.get('obs_count', -1))
                        if sp_oc != 1:
                            issues.append(
                                f"[{SHIPS_GROUP_NAME}] obs_count={sp_oc}, "
                                f"expected 1 (one cycle-level parameter set)."
                            )
                    except (ValueError, TypeError):
                        pass

                    # 19c: All expected predictor datasets must be present
                    sp_vars = set(k for k in sp.keys()
                                 if isinstance(sp[k], h5py.Dataset))
                    missing_preds = SHIPS_EXPECTED_DATASETS - sp_vars
                    if missing_preds:
                        issues.append(
                            f"[{SHIPS_GROUP_NAME}] "
                            f"{len(missing_preds)} expected predictor(s) "
                            f"missing: {', '.join(sorted(missing_preds))}"
                        )

                    # 19d: Per-dataset checks
                    for col in sp_vars:
                        dset = sp[col]
                        # Shape must be (1,)
                        if dset.shape != (1,):
                            issues.append(
                                f"[{SHIPS_GROUP_NAME}/{col}] "
                                f"Shape {dset.shape}, expected (1,)."
                            )
                        # dtype must be float64
                        if (np.issubdtype(dset.dtype, np.number)
                                and dset.dtype != np.float64):
                            issues.append(
                                f"[{SHIPS_GROUP_NAME}/{col}] Advisory: "
                                f"dtype is {dset.dtype}, expected float64."
                            )
                        # units attribute must be present
                        if 'units' not in dset.attrs:
                            issues.append(
                                f"[{SHIPS_GROUP_NAME}/{col}] "
                                f"Missing 'units' attribute."
                            )
                        # long_name attribute must be present
                        if 'long_name' not in dset.attrs:
                            issues.append(
                                f"[{SHIPS_GROUP_NAME}/{col}] "
                                f"Missing 'long_name' attribute."
                            )

    except Exception as e:
        issues.append(f"[FATAL] Could not read file: {e}")

    return issues, summary


# =============================================================================
# RUNNER
# =============================================================================

def run_validation(target_dir, issues_report="validation_issues.csv",
                   counts_report="validation_obs_counts.csv"):
    """
    Validate all AI-ready HDF5 files in target_dir.

    All per-file issues are written to issues_report (CSV).
    Obs counts per group per file are written to counts_report (CSV).
    Only a concise pass/fail summary is printed to the screen.
    """
    print(f"🔍 Strict AI-Ready Validation (v9)")
    print(f"   Directory : {target_dir}")
    print(f"   Basin     : {_BASIN_LABEL.get(BASIN_FILTER, BASIN_FILTER)}")
    print(f"   Sentinels : {SENTINEL_VALUES}")
    print()

    h5_files = sorted(
        glob.glob(os.path.join(target_dir, "**", "*.hdf5"), recursive=True)
    )
    if not h5_files:
        print("❌ No .hdf5 files found.")
        return

    # Apply basin filter
    all_count = len(h5_files)
    h5_files = [f for f in h5_files if should_validate_file(f)]
    n_skipped = all_count - len(h5_files)
    if n_skipped > 0:
        print(f"   Skipped {n_skipped} file(s) outside "
              f"{_BASIN_LABEL.get(BASIN_FILTER, BASIN_FILTER)} basin.")
    print(f"   Validating {len(h5_files)} file(s).\n")

    total_files   = len(h5_files)
    failed_files  = 0
    all_summaries = []
    all_issues    = []   # accumulated for CSV write

    for i, filepath in enumerate(h5_files):
        issues, summary = validate_file(filepath)
        all_summaries.append(summary)

        if issues:
            failed_files += 1
            for issue in issues:
                severity = "Advisory" if "Advisory" in issue else "Error"
                all_issues.append({
                    "Filename": os.path.basename(filepath),
                    "Severity": severity,
                    "Issue":    issue,
                })

        if (i + 1) % 500 == 0:
            print(f"   [Progress] {i+1}/{total_files} files validated...")

    # --------------------------------------------------------------------------
    # CROSS-FILE CHECK: storm_id naming convention
    #
    # A file whose storm_id uses a number-word or INVEST prefix is only a
    # problem when a proper name exists for the same (year, nhc_code) group —
    # i.e. a double-entry situation where renaming should have been applied.
    # Files that are legitimately the only name for their NHC code are fine.
    # --------------------------------------------------------------------------
    # Build (year, nhc_code) -> set of storm_ids from all filenames
    from collections import defaultdict
    group_ids = defaultdict(set)   # (year, nhc_code) -> {storm_id, ...}
    file_meta = {}                 # filename -> (year, nhc_code, storm_id)

    for fp in h5_files:
        fname = os.path.basename(fp)
        m = re.search(r'HRDOBS_([A-Z0-9]+)[._](\d{4})(\d{2})(\d{2})', fname)
        if not m:
            continue
        storm_id = m.group(1)
        year     = int(m.group(2))
        nhc_m    = re.search(r'(\d+[A-Z]+)$', storm_id)
        if not nhc_m:
            continue
        nhc_code = nhc_m.group(1)
        group_ids[(year, nhc_code)].add(storm_id)
        file_meta[fname] = (year, nhc_code, storm_id)

    # For each group with multiple names, flag files whose storm_id is a
    # lower-tier placeholder when a higher-tier name exists in the same group
    def name_tier(sid):
        mp = re.match(r'([A-Za-z]+)', sid)
        p  = mp.group(1).upper() if mp else ''
        if p in _NHC_SPELLED_NUMBERS: return 2   # lowest
        if p in _NHC_INVEST:          return 1   # middle
        return 0                                  # proper name (best)

    naming_issues = []
    for (year, nhc_code), sids in group_ids.items():
        if len(sids) < 2:
            continue   # single name — no problem regardless of what it is
        best_tier = min(name_tier(s) for s in sids)
        for fname, (fy, fn, fid) in file_meta.items():
            if fy != year or fn != nhc_code:
                continue
            if name_tier(fid) > best_tier:
                category = ('INVEST placeholder' if fid.upper().startswith('INVEST')
                            else 'number-word placeholder')
                naming_issues.append({
                    "Filename": fname,
                    "Severity": "Error",
                    "Issue": (
                        f"[GLOBAL] storm_id '{fid}' uses a {category} prefix "
                        f"but a higher-tier name exists for {year} {nhc_code} "
                        f"({', '.join(sorted(sids - {fid}))}) — file should "
                        f"have been renamed before conversion."
                    ),
                })

    if naming_issues:
        # Count distinct filenames affected
        affected = len({r['Filename'] for r in naming_issues})
        failed_files += affected
        all_issues.extend(naming_issues)

    # --------------------------------------------------------------------------
    # Write issues report
    # --------------------------------------------------------------------------
    if all_issues:
        pd.DataFrame(all_issues).to_csv(issues_report, index=False)

    # --------------------------------------------------------------------------
    # Screen summary — concise
    # --------------------------------------------------------------------------
    print()
    print("=" * 60)
    if failed_files == 0:
        print(f"✅ PASSED — all {total_files} files passed all checks.")
    else:
        error_count   = sum(1 for r in all_issues if r["Severity"] == "Error")
        advisor_count = sum(1 for r in all_issues if r["Severity"] == "Advisory")
        print(f"⚠️  FAILED — {failed_files}/{total_files} files had issues.")
        print(f"   Hard errors : {error_count}")
        print(f"   Advisories  : {advisor_count}")
        print(f"   Details     → {issues_report}")
    print("=" * 60)

    # --------------------------------------------------------------------------
    # Obs-count report
    # --------------------------------------------------------------------------
    count_rows = []
    for s in all_summaries:
        for grp, cnt in s['obs_counts'].items():
            count_rows.append({
                'Filename':  s['filename'],
                'Group':     grp,
                'Obs_Count': cnt,
            })

    if count_rows:
        df_counts = pd.DataFrame(count_rows)
        df_counts.to_csv(counts_report, index=False)

        summary_stats = (
            df_counts[df_counts['Obs_Count'] > 0]
            .groupby('Group')['Obs_Count']
            .agg(Files='count', Min='min', Median='median',
                 Max='max', Total='sum')
            .reset_index()
            .sort_values('Total', ascending=False)
        )
        print(f"\n📊 Obs-count report → {counts_report}")
        print()
        print("Per-group obs summary across all files:")
        print(summary_stats.to_string(index=False))

    # --------------------------------------------------------------------------
    # Spline track pressure summary
    # --------------------------------------------------------------------------
    pres_values = []
    pres_missing = 0
    pres_present = 0
    spline_total = 0

    for fp in h5_files:
        try:
            with h5py.File(fp, 'r') as f:
                if SPLINE_TRACK_GROUP not in f:
                    continue
                grp = f[SPLINE_TRACK_GROUP]
                if not isinstance(grp, h5py.Group):
                    continue
                oc = grp.attrs.get('obs_count', 0)
                try:
                    oc = int(oc)
                except (ValueError, TypeError):
                    oc = 0
                if oc <= 0:
                    continue

                spline_total += 1
                if 'pres' in grp:
                    pres_present += 1
                    data = grp['pres'][:]
                    non_nan = data[~np.isnan(data)]
                    if len(non_nan) > 0:
                        pres_values.append(float(non_nan[0]))
                else:
                    pres_missing += 1
        except Exception:
            continue

    if spline_total > 0:
        print(f"\n📊 Spline Track Pressure Summary")
        print(f"   Files with spline track  : {spline_total}")
        print(f"   With pres dataset        : {pres_present}")
        print(f"   Missing pres dataset     : {pres_missing}")
        if pres_values:
            pres_arr = np.array(pres_values)
            print(f"\n   Pressure distribution (n={len(pres_values)}):")
            # Bin by standard flight levels
            from collections import Counter
            rounded = [round(v / 50.0) * 50 for v in pres_values]
            counts = Counter(rounded)
            for lvl in sorted(counts.keys()):
                cnt = counts[lvl]
                bar = '█' * (cnt // 5) + ('▌' if cnt % 5 >= 3 else '')
                print(f"     {lvl:6.0f} hPa : {cnt:4d}  {bar}")


    # --------------------------------------------------------------------------
    # ships_params group summary (v9)
    # --------------------------------------------------------------------------
    ships_present  = 0
    ships_absent   = 0
    pred_valid     = {col: 0 for col in SHIPS_EXPECTED_DATASETS}
    pred_total     = {col: 0 for col in SHIPS_EXPECTED_DATASETS}

    for fp in h5_files:
        try:
            with h5py.File(fp, 'r') as f:
                if SHIPS_GROUP_NAME not in f:
                    ships_absent += 1
                    continue
                sp = f[SHIPS_GROUP_NAME]
                if not isinstance(sp, h5py.Group):
                    ships_absent += 1
                    continue
                ships_present += 1
                for col in SHIPS_EXPECTED_DATASETS:
                    if col not in sp:
                        continue
                    dset = sp[col]
                    if not isinstance(dset, h5py.Dataset):
                        continue
                    pred_total[col] += 1
                    data = dset[:]
                    if data.size > 0 and not np.all(np.isnan(data)):
                        pred_valid[col] += 1
        except Exception:
            continue

    if ships_present + ships_absent > 0:
        print(f"\n📊 SHIPS Parameters Summary")
        print(f"   Files with ships_params  : {ships_present} "
              f"({100*ships_present/(ships_present+ships_absent):.1f}%)")
        print(f"   Files without            : {ships_absent}")
        if ships_present > 0:
            print(f"")
            print(f"   Per-predictor coverage (non-NaN) across "
                  f"{ships_present} files with ships_params:")
            print(f"   {'Predictor':<22}  {'Valid':>6}  {'Total':>6}  {'Pct':>6}")
            print(f"   {'-'*22}  {'-'*6}  {'-'*6}  {'-'*6}")
            for col in sorted(SHIPS_EXPECTED_DATASETS):
                n_v = pred_valid.get(col, 0)
                n_t = pred_total.get(col, 0)
                pct = 100.0 * n_v / n_t if n_t > 0 else 0.0
                print(f"   {col:<22}  {n_v:>6,}  {n_t:>6,}  {pct:>5.1f}%")


if __name__ == "__main__":
    TARGET_DIR = "AI_ready_dataset/HRDOBS_hdf5"
    run_validation(TARGET_DIR)
    