import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

# Set font to Arial Narrow
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial Narrow', 'Arial', 'Helvetica']

# CRITICAL FIX: Ensure fonts are exported as editable text objects (Type 42), not individual paths (Type 3)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Set to save figure
do_fig_save = True

# =============================================================================
# 1. I/O DEFINITIONS
# =============================================================================
INPUT_HRDOBS_DB_FILE = 'hrdobs_inventory_db.csv'
OUTPUT_FIGURE_FILE   = 'hrdobs_figure7.pdf'

# =========================================================================
# 2. DYNAMIC DATA EXTRACTION 
# =========================================================================
db_path = INPUT_HRDOBS_DB_FILE
group_totals = {}

try:
    df = pd.read_csv(db_path)
    # Parse the JSON string in each row to accumulate the total observations per group
    for count_str in df['Group_Counts_JSON'].dropna():
        try:
            counts = json.loads(count_str)
            for grp, val in counts.items():
                group_totals[grp] = group_totals.get(grp, 0) + val
        except Exception:
            pass
except FileNotFoundError:
    print(f"Error: Could not find '{db_path}'. Make sure it is in the same directory.")
    exit()

def get_tot(grp_name):
    """Safely fetch the total for a group, returning 0 if it doesn't exist."""
    return group_totals.get(grp_name, 0)

# =========================================================================

# Map colors
c_p3  = 'navy'       # NOAA P-3
c_g4  = '#66FF66'    # NOAA G-IV (bright light green)
c_af  = '#B0B0B0'    # Air Force (medium light gray)
c_gh  = 'darkred'    # NASA Global Hawk
c_trk = '#FFFF99'    # Tracks (light yellow)

# Assemble data groups as (Value, Color)
# DC-8 and missing G-IV SFMR are omitted cleanly
groups_data = [
    # Group 1 - Dropsondes (P-3, G-IV, AF, Global Hawk)
    [(get_tot('dropsonde_noaa42') + get_tot('dropsonde_noaa43'), c_p3), 
     (get_tot('dropsonde_noaa49'), c_g4), 
     (get_tot('dropsonde_usaf'), c_af), 
     (get_tot('dropsonde_ghawk'), c_gh)],
     
    # Group 2 - Flight Level (P-3, G-IV, AF)
    [(get_tot('flight_level_hdobs_noaa42') + get_tot('flight_level_hdobs_noaa43'), c_p3), 
     (get_tot('flight_level_hdobs_noaa49'), c_g4), 
     (get_tot('flight_level_hdobs_usaf'), c_af)],
     
    # Group 3 - SFMR (P-3, AF)
    [(get_tot('sfmr_noaa42') + get_tot('sfmr_noaa43'), c_p3), 
     (get_tot('sfmr_usaf'), c_af)],
     
    # Group 4 - TDR (P-3, G-IV)
    [(get_tot('tdr_noaa42') + get_tot('tdr_noaa43'), c_p3), 
     (get_tot('tdr_noaa49'), c_g4)],
     
    # Group 5 - SHIPS Environment (Mapped with track color)
    [(get_tot('ships_params'), c_trk)],
     
    # Group 6 - Vortex Messages
    [(get_tot('track_vortex_message'), c_trk)],
    
    # Group 7 - Best Track
    [(get_tot('track_best_track'), c_trk)],
    
    # Group 8 - High-resolution Track
    [(get_tot('track_spline_track'), c_trk)]
]

group_labels = [
    'Dropsondes', 
    'Flight Level', 
    'SFMR', 
    'TDR', 
    'SHIPS',
    'Vortex Msg.', 
    'Best Track', 
    'High-Res. Track'
]

# Set Figure Size: 5 inches wide, 3.5 inches tall
fig, ax = plt.subplots(figsize=(5, 3.5))
fig.subplots_adjust(left=0.15, bottom=0.28, right=0.95, top=0.95)

# Layout parameters
barWidth = 0.35    
groupGap = 0.8     
y_min = 1000  # Explicit bottom for log scale to prevent paths dropping to 0

groupCenters = []
currX = 0

# Plotting loop
for g_idx, bars in enumerate(groups_data):
    k = len(bars)
    if k == 0:
        currX += groupGap
        groupCenters.append(np.nan)
        continue

    # x positions for bars in this group
    xi = currX + np.arange(k) * barWidth
    groupCenters.append(np.mean(xi))

    # plot bars for this group, applying the exact mapped color and the y_min base
    for j, (val, color) in enumerate(bars):
        if np.isnan(val) or val <= 0:
            continue
        
        ax.bar(xi[j], val, width=barWidth, bottom=y_min, color=color, edgecolor='black', linewidth=1.2, zorder=3)

    currX += k * barWidth + groupGap

# Log scale and formatting
ax.set_yscale('log')

# Explicitly set the visible y-limit to match our bar base
ax.set_ylim(bottom=y_min)

ax.grid(axis='y', linestyle='-', zorder=0)  # Y-grid on
ax.grid(axis='x', visible=False)            # X-grid off
ax.tick_params(direction='out')

# Set ticks and Labels
ax.set_xticks(groupCenters)
ax.set_xticklabels(group_labels, rotation=45, ha='right')
ax.set_ylabel('Number of Observations')

ax.set_xlim(-barWidth, currX - groupGap + barWidth)

# Save as vector
if do_fig_save:
    plt.savefig(OUTPUT_FIGURE_FILE, format='pdf')
