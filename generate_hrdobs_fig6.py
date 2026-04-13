import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# =============================================================================
# 1. I/O DEFINITIONS
# =============================================================================
INPUT_HURDAT_FILE    = 'HURDAT2_all_since_2014_filtered.csv'
INPUT_HRDOBS_DB_FILE = 'hrdobs_inventory_db.csv'
OUTPUT_FIGURE_FILE   = 'hrdobs_figure6.pdf'

# =============================================================================
# 2. CONFIGURATION & STYLING
# =============================================================================

# Force Matplotlib to export text as editable font objects in Illustrator
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.sans-serif'] = ['Arial Narrow']
mpl.rcParams['font.family'] = 'sans-serif'

# Storm intensity colors (from established conventions)
COLOR_MAP = {
    'TD': (66/255,  199/255, 244/255),
    'TS': (17/255,  170/255, 75/255),
    'H1': (17/255,  170/255, 75/255),
    'H2': (254/255, 230/255, 9/255),
    'H3': (254/255, 230/255, 9/255),
    'H4': (239/255, 61/255,  37/255),
    'H5': (239/255, 61/255,  37/255),
    'EX': (238/255, 193/255, 219/255), 
}

# The specific stacking order for the bars (bottom to top)
CATEGORY_ORDER = ['TD', 'TS', 'H1', 'H2', 'H3', 'H4', 'H5', 'EX']

# Formatting Constants
BAR_EDGE_COLOR = 'black'  # Updated to solid black per your request
BAR_LINE_WIDTH = 2.0

# =============================================================================
# 3. DYNAMIC DATA LOADING & CROSS-REFERENCING
# =============================================================================

# Load the base HURDAT dataset (utilizing the existing header row)
df = pd.read_csv(INPUT_HURDAT_FILE)
df['CAT'] = df['CAT'].astype(str).str.strip().str.upper()

# Construct a unique temporal key for HURDAT (Format: NAME_YYYYMMDDHH)
hurdat_name = df['NAME'].astype(str).str.strip().str.upper()
hurdat_yr   = df['YEAR'].astype(str)
hurdat_mo   = df['MONTH'].astype(str).str.zfill(2)
hurdat_dy   = df['DAY'].astype(str).str.zfill(2)
hurdat_hr   = (df['HOUR'] // 100).astype(str).str.zfill(2)

df['Match_Key'] = hurdat_name + '_' + hurdat_yr + hurdat_mo + hurdat_dy + hurdat_hr

# Load the HRDOBS inventory database
try:
    hrdobs_db = pd.read_csv(INPUT_HRDOBS_DB_FILE)
    
    # Construct the identical matching key for HRDOBS
    hrdobs_name   = hrdobs_db['Storm'].astype(str).str.strip().str.upper()
    hrdobs_dt     = pd.to_datetime(hrdobs_db['Storm_Datetime'])
    hrdobs_dt_str = hrdobs_dt.dt.strftime('%Y%m%d%H')
    
    hrdobs_keys = set(hrdobs_name + '_' + hrdobs_dt_str)
    
except FileNotFoundError:
    print(f"Error: Could not find '{INPUT_HRDOBS_DB_FILE}'. Ensure it is in the same directory.")
    hrdobs_keys = set()

# Cross-reference to identify which HURDAT rows have AI-Ready HRDOBS data
df['HRDOBS'] = df['Match_Key'].isin(hrdobs_keys).astype(int)

# Filter out unwanted categories
unwanted_cats = ['ET', 'LO', 'DB']
df = df[~df['CAT'].isin(unwanted_cats)]

hrdobs_df = df[df['HRDOBS'] == 1]

# =============================================================================
# 4. FIGURE & AXES SETUP (Exact Artboard Placement)
# =============================================================================

fig_width_in  = 12.50
fig_height_in = 6.37
plot_size_in  = 5.00

# Calculate normalized coordinates (0 to 1) for Matplotlib
left_margin_in = 1.00
gap_in         = 1.00
bottom_margin_in = 0.80

ax1_left = left_margin_in / fig_width_in
ax2_left = (left_margin_in + plot_size_in + gap_in) / fig_width_in
bottom   = bottom_margin_in / fig_height_in
width    = plot_size_in / fig_width_in
height   = plot_size_in / fig_height_in

fig = plt.figure(figsize=(fig_width_in, fig_height_in))
ax1 = fig.add_axes([ax1_left, bottom, width, height]) # Panel A
ax2 = fig.add_axes([ax2_left, bottom, width, height]) # Panel B

# Apply universal box and tick styling to both axes
for ax in [ax1, ax2]:
    ax.tick_params(axis='both', direction='in', width=3.0, length=8.0, 
                   labelsize=15, top=True, right=True, pad=6)
    for spine in ax.spines.values():
        spine.set_linewidth(3.0)
        spine.set_zorder(10)

# =============================================================================
# 5. PREPARE STACKED HISTOGRAM DATA
# =============================================================================

# Extract unique years and set up our X-axis ticks (every 2 years from 2014)
min_year = 2014
max_year = int(df['YEAR'].max())
x_ticks = list(range(min_year, max_year + 1, 2))

# Helper function to aggregate data into a stacking-ready format
def prepare_stacked_data(data):
    counts = data.groupby(['YEAR', 'CAT']).size().unstack(fill_value=0)
    # Ensure all categories exist in the dataframe columns, even if count is 0
    for cat in CATEGORY_ORDER:
        if cat not in counts.columns:
            counts[cat] = 0
    return counts[CATEGORY_ORDER] # Enforce column order for stacking

counts_all = prepare_stacked_data(df)
counts_hrd = prepare_stacked_data(hrdobs_df)

years_all = counts_all.index.values
years_hrd = counts_hrd.index.values

# =============================================================================
# 6. PLOT PANEL A: ALL BEST TRACK CYCLES
# =============================================================================

bottoms_all = np.zeros(len(years_all))

for cat in CATEGORY_ORDER:
    ax1.bar(years_all, counts_all[cat], bottom=bottoms_all, 
            color=COLOR_MAP.get(cat), edgecolor=BAR_EDGE_COLOR, 
            linewidth=BAR_LINE_WIDTH, zorder=3)
    bottoms_all += counts_all[cat].values

# Formatting limits and ticks based on MATLAB script logic
ax1.set_ylim([0, 750])
ax1.set_yticks(range(0, 751, 150))
ax1.grid(axis='y', linestyle='-', linewidth=0.5, color='gray', alpha=0.5, zorder=0)

ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_ticks, rotation=45, ha='right', rotation_mode="anchor")

# Titles & Labels
ax1.set_title('(a) All Best Track Cycles', fontsize=27, fontname='Arial Narrow', pad=15)
ax1.set_ylabel('Number of Cycles', fontsize=20, fontname='Arial Narrow', labelpad=10)
ax1.set_xlabel('Hurricane Season', fontsize=20, fontname='Arial Narrow', labelpad=10)

# =============================================================================
# 7. PLOT PANEL B: HRDOBS CYCLES ONLY
# =============================================================================

bottoms_hrd = np.zeros(len(years_hrd))

for cat in CATEGORY_ORDER:
    ax2.bar(years_hrd, counts_hrd[cat], bottom=bottoms_hrd, 
            color=COLOR_MAP.get(cat), edgecolor=BAR_EDGE_COLOR, 
            linewidth=BAR_LINE_WIDTH, zorder=3)
    bottoms_hrd += counts_hrd[cat].values

# Formatting limits and ticks based on MATLAB script logic
ax2.set_ylim([0, 275])
ax2.set_yticks(range(0, 276, 55))
ax2.grid(axis='y', linestyle='-', linewidth=0.5, color='gray', alpha=0.5, zorder=0)

ax2.set_xticks(x_ticks)
ax2.set_xticklabels(x_ticks, rotation=45, ha='right', rotation_mode="anchor")

# Titles & Labels
ax2.set_title('(b) HRDOBS Cycles Only', fontsize=27, fontname='Arial Narrow', pad=15)
ax2.set_ylabel('Number of Cycles', fontsize=20, fontname='Arial Narrow', labelpad=10)
ax2.set_xlabel('Hurricane Season', fontsize=20, fontname='Arial Narrow', labelpad=10)

# =============================================================================
# 8. EXPORT
# =============================================================================

output_file = OUTPUT_FIGURE_FILE
plt.savefig(output_file, format='pdf', facecolor='white', edgecolor='none')

print(f"✅ Figure 6 generated successfully: {output_file}")
