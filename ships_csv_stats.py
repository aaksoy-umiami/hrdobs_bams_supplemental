# -*- coding: utf-8 -*-
"""
ships_csv_stats.py
------------------
Reads a SHIPS CSV file produced by ships_to_csv.py and prints a structured
statistical summary to the terminal, then writes the same summary to a
companion .txt report file.

Usage
-----
    python ships_csv_stats.py <ships_csv_file> [output_report_file]

If output_report_file is omitted, the report is written alongside the CSV
with the same base name and a _stats.txt extension.
"""

import sys
import os
import csv
from pathlib import Path
from collections import defaultdict


# =============================================================================
# CONFIGURATION
# =============================================================================

META_COLS = {
    'storm_name', 'atcf_id', 'hrdobs_storm_id',
    'year', 'month', 'day', 'hour', 'datetime_utc',
    'vmax_hd_kt', 'lat_hd_degN', 'lon_hd_degE', 'mslp_hd_hpa',
}

_BASIN_SUFFIX = {
    'AL': 'L', 'EP': 'E', 'CP': 'C',
    'WP': 'W', 'IO': 'I', 'SH': 'S',
}

COL_LABELS = {
    'type':            'Storm type (categorical)',
    'incv_kt':         'Intensity change -6 to 0 hr (kt)',
    'csst_degc':       'Climatological SST (deg C)',
    'cd20_m':          'Clim. depth of 20 deg C isotherm (m)',
    'cd26_m':          'Clim. depth of 26 deg C isotherm (m)',
    'cohc_kjcm2':      'Clim. ocean heat content (kJ/cm2)',
    'dtl_km':          'Distance to land (km)',
    'oage_hr':         'Ocean age (hr)',
    'nage_hr':         'Intensity-weighted ocean age (hr)',
    'shrd_kt':         '850-200 hPa shear magnitude (kt)',
    'shtd_deg':        '850-200 hPa shear heading (deg)',
    'shdc_kt':         'Vortex-removed shear magnitude (kt)',
    'sddc_deg':        'Vortex-removed shear heading (deg)',
    'rhlo_pct':        '850-700 hPa relative humidity (%)',
    'rhmd_pct':        '700-500 hPa relative humidity (%)',
    'rhhi_pct':        '500-300 hPa relative humidity (%)',
    'vmpi_kt':         'Max potential intensity (kt)',
    'penv_hpa':        'Environmental surface pressure (hPa)',
    'penc_hpa':        'Outer vortex surface pressure (hPa)',
    'z850_1e7_per_s':  '850 hPa vorticity (10^-7 s^-1)',
    'd200_1e7_per_s':  '200 hPa divergence (10^-7 s^-1)',
    'u200_kt':         '200 hPa zonal wind (kt)',
    'dsst_degc':       'Daily Reynolds SST (deg C)',
    'nsst_degc':       'NCODA analysis SST (deg C)',
    'nohc_kjcm2':      'NCODA ocean heat content (kJ/cm2)',
}

TYPE_NAMES = {
    '0': 'Wave / Remnant Low / Dissipating',
    '1': 'Tropical',
    '2': 'Subtropical',
    '3': 'Extratropical',
}


# =============================================================================
# HELPERS
# =============================================================================

def nhc_id_from_atcf(atcf_id):
    """Convert 'AL092014' -> '09L'."""
    try:
        basin  = atcf_id[:2].upper()
        number = atcf_id[2:4]
        suffix = _BASIN_SUFFIX.get(basin, '?')
        return f"{number}{suffix}"
    except Exception:
        return '??'


def load_csv(path):
    """Load the SHIPS CSV, skipping # comment lines."""
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(
            line for line in fh if not line.startswith('#')
        )
        return list(reader)


def is_valid(value):
    return value is not None and str(value).strip() != ''


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# =============================================================================
# STATISTICS COMPUTATION
# =============================================================================

def compute_stats(rows):
    total     = len(rows)
    pred_cols = [c for c in rows[0].keys() if c not in META_COLS]

    # Year-level
    years_present  = sorted(set(int(r['year']) for r in rows if is_valid(r.get('year'))))
    cycles_by_year = defaultdict(int)
    storms_by_year = defaultdict(set)
    for r in rows:
        yr = int(r['year'])
        cycles_by_year[yr] += 1
        storms_by_year[yr].add(r['atcf_id'])

    # Storm-level
    unique_storms = defaultdict(lambda: {
        'nhc_id': '', 'storm_name': '', 'years': set(), 'cycles': 0
    })
    for r in rows:
        aid = r['atcf_id']
        unique_storms[aid]['nhc_id']     = nhc_id_from_atcf(aid)
        unique_storms[aid]['storm_name'] = r.get('storm_name', '')
        unique_storms[aid]['years'].add(int(r['year']))
        unique_storms[aid]['cycles'] += 1

    # Type distribution
    type_dist = defaultdict(int)
    for r in rows:
        t = str(r.get('type', '')).strip()
        type_dist[t if t else 'missing'] += 1

    # Per-predictor coverage + basic stats
    predictor_stats = {}
    for col in pred_cols:
        values = [safe_float(r.get(col)) for r in rows]
        valid  = [v for v in values if v is not None]
        n      = len(valid)
        predictor_stats[col] = {
            'n_valid': n,
            'n_total': total,
            'pct':     100.0 * n / total if total > 0 else 0.0,
            'min':     min(valid)     if valid else None,
            'max':     max(valid)     if valid else None,
            'mean':    sum(valid) / n if valid else None,
        }

    return {
        'total_cycles':    total,
        'years_present':   years_present,
        'cycles_by_year':  dict(cycles_by_year),
        'storms_by_year':  dict(storms_by_year),
        'unique_storms':   dict(unique_storms),
        'type_dist':       dict(type_dist),
        'predictor_stats': predictor_stats,
        'pred_cols':       pred_cols,
    }


# =============================================================================
# REPORT FORMATTING
# =============================================================================

W = 72

def _hdr(title):
    return ['', '=' * W, title, '=' * W]

def _sub(title):
    return ['', f'── {title} ' + '─' * max(0, W - len(title) - 4)]

def _fmt(v, dec=1):
    if v is None:
        return ' ' * 9
    return f"{v:.{dec}f}"[:9].rjust(9)


def format_report(stats, csv_path):
    lines = []
    add = lines.append
    ext = lines.extend

    total     = stats['total_cycles']
    years     = stats['years_present']
    by_year   = stats['cycles_by_year']
    storms_yr = stats['storms_by_year']
    storms    = stats['unique_storms']
    type_dist = stats['type_dist']
    pstats    = stats['predictor_stats']
    pred_cols = stats['pred_cols']

    # ── 1. File Overview ─────────────────────────────────────────────────────
    ext(_hdr('SHIPS CSV Statistics Report'))
    add(f'  Source file     : {csv_path}')
    add('')
    ext(_sub('1. File Overview'))
    add(f'  Total cycles    : {total:,}')
    add(f'  Year range      : {years[0]} \u2013 {years[-1]}')
    add(f'  Number of years : {len(years)}')
    add(f'  Years present   : {", ".join(str(y) for y in years)}')
    add(f'  Unique storms   : {len(storms):,}  (by ATCF ID, e.g. AL092014)')

    # ── 2. Storms and Cycles per Year ────────────────────────────────────────
    ext(_sub('2. Storms and Cycles per Year'))
    add(f'  {"Year":<6}  {"Storms":>8}  {"Cycles":>8}')
    add(f'  {"----":<6}  {"------":>8}  {"------":>8}')
    for yr in years:
        n_s = len(storms_yr.get(yr, set()))
        n_c = by_year.get(yr, 0)
        add(f'  {yr:<6}  {n_s:>8,}  {n_c:>8,}')
    add(f'  {"TOTAL":<6}  {len(storms):>8,}  {total:>8,}')

    # ── 3. Storm Type Distribution ───────────────────────────────────────────
    ext(_sub('3. Storm Type Distribution'))
    add(f'  {"Type":<10}  {"Cycles":>8}  {"Pct":>7}  Description')
    add(f'  {"----":<10}  {"------":>8}  {"---":>7}  -----------')
    for t_key in sorted(type_dist.keys()):
        count = type_dist[t_key]
        label = TYPE_NAMES.get(t_key, f'Unknown ({t_key})')
        pct   = 100.0 * count / total if total > 0 else 0.0
        add(f'  {t_key:<10}  {count:>8,}  {pct:>6.1f}%  {label}')

    # ── 4. Unique Storm Identifiers ──────────────────────────────────────────
    ext(_sub('4. Unique Storm Identifiers'))
    add('  NHC-style format: <number><basin_suffix>, e.g. 09L = 9th Atlantic storm.')
    add('  The same NHC number in different years is a different storm.')
    add('')
    add(f'  {"NHC ID":<7}  {"Cycles":>7}  {"Year(s)"}')
    add(f'  {"------":<7}  {"------":>7}  {"-------"}')

    def _sort_key(atcf_id):
        d   = storms[atcf_id]
        num = d['nhc_id'][:2] if len(d['nhc_id']) >= 2 else '99'
        sfx = d['nhc_id'][2:] if len(d['nhc_id']) > 2 else ''
        return (num, sfx, min(d['years']))

    for atcf_id in sorted(storms.keys(), key=_sort_key):
        d      = storms[atcf_id]
        nhc    = d['nhc_id']
        n_cyc  = d['cycles']
        yr_str = ', '.join(str(y) for y in sorted(d['years']))
        add(f'  {nhc:<7}  {n_cyc:>7,}  {yr_str}')

    # ── 5. Predictor Coverage ────────────────────────────────────────────────
    ext(_sub('5. Predictor Coverage at t=0'))
    add(
        f'  {"Column":<22}  {"Valid":>6}  {"Total":>6}  {"Pct":>6}  '
        f'{"Min":>9}  {"Mean":>9}  {"Max":>9}  Description'
    )
    add(
        f'  {"-"*22}  {"-"*6}  {"-"*6}  {"-"*6}  '
        f'{"-"*9}  {"-"*9}  {"-"*9}  {"-"*30}'
    )
    for col in pred_cols:
        ps    = pstats[col]
        label = COL_LABELS.get(col, col)
        add(
            f'  {col:<22}  {ps["n_valid"]:>6,}  {ps["n_total"]:>6,}  '
            f'{ps["pct"]:>5.1f}%  '
            f'{_fmt(ps["min"])}  {_fmt(ps["mean"])}  {_fmt(ps["max"])}  '
            f'{label}'
        )

    add('')
    add('=' * W)
    return '\n'.join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python ships_csv_stats.py <ships_csv_file> [output_report_file]")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.isfile(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    report_path = sys.argv[2] if len(sys.argv) >= 3 else \
                  str(Path(csv_path).with_name(Path(csv_path).stem + '_stats.txt'))

    print(f"Reading: {csv_path}")
    rows = load_csv(csv_path)
    if not rows:
        print("No data rows found (file may be empty or all comments).")
        sys.exit(1)
    print(f"  Rows loaded: {len(rows):,}")

    stats  = compute_stats(rows)
    report = format_report(stats, csv_path)

    print()
    print(report)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report + '\n')
    print(f"\nReport written -> {report_path}")


if __name__ == '__main__':
    main()
