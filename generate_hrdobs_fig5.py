import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# =============================================================================
# 1. I/O DEFINITIONS
# =============================================================================
INPUT_HURDAT_FILE    = 'HURDAT2_all_since_2014_filtered.csv'
INPUT_HRDOBS_DB_FILE = 'hrdobs_inventory_db.csv'
OUTPUT_FIGURE_FILE   = 'hrdobs_figure5.pdf'

# =============================================================================
# 2. CONFIGURATION & STYLING
# =============================================================================

# Force Matplotlib to export text as editable font objects in Illustrator
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.sans-serif'] = ['Arial Narrow']
mpl.rcParams['font.family'] = 'sans-serif'

# Storm intensity colors
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

CATEGORY_ORDER = ['TD', 'TS', 'H1', 'H2', 'H3', 'H4', 'H5', 'EX']
PLOT_ORDER = ['EX', 'TD', 'TS', 'H1', 'H2', 'H3', 'H4', 'H5']

# Formatting Constants
LINE_WIDTH = 3.0
SCATTER_SIZE = 15
MARKER_EDGE_WIDTH = 0.4

# Custom line styling for Regression Lines
REG_COLOR_ALL = (150/255, 150/255, 150/255) # Gray
REG_STYLE_HRD = (0, (2, 2)) # 2pt dash, 2pt gap

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

# Create the specific subset for plotting
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
# 5. PANEL A: INTENSITY HISTOGRAM
# =============================================================================

total_counts = df['CAT'].value_counts(normalize=True) * 100
hrdobs_counts = hrdobs_df['CAT'].value_counts(normalize=True) * 100

all_vals = [total_counts.get(cat, 0) for cat in CATEGORY_ORDER]
hrdobs_vals = [hrdobs_counts.get(cat, 0) for cat in CATEGORY_ORDER]

colors = [COLOR_MAP.get(cat, (0.5, 0.5, 0.5)) for cat in CATEGORY_ORDER]

# Plot Bars
ax1.bar(CATEGORY_ORDER, all_vals, color=colors, edgecolor=(0.75, 0.75, 0.75), 
        linewidth=1.5, zorder=2)
ax1.bar(CATEGORY_ORDER, hrdobs_vals, facecolor='none', edgecolor='black', 
        linestyle='--', linewidth=2.0, zorder=3)

# Formatting
ax1.set_ylim([0, 50])
ax1.set_yticks(range(0, 51, 10))
ax1.grid(axis='y', linestyle='-', linewidth=0.5, color='gray', alpha=0.5, zorder=0)

# Panel A Titles and Labels
ax1.set_title('(a) Intensity Distribution', fontsize=27, fontname='Arial Narrow', pad=15)
ax1.set_ylabel('Normalized Number of Cycles (% of Total)', fontsize=20, fontname='Arial Narrow', labelpad=10)
ax1.set_xlabel('Storm Category', fontsize=20, fontname='Arial Narrow', labelpad=10)

# Calculate TS Annotations for Terminal Output
total_hurdat = len(df)
total_ts_hurdat = len(df[df['CAT'] == 'TS'])
total_hrdobs = len(hrdobs_df)
total_ts_hrdobs = len(hrdobs_df[hrdobs_df['CAT'] == 'TS'])

# =============================================================================
# 6. PANEL B: WIND-PRESSURE RELATIONSHIP
# =============================================================================

x_all = df['INT'].values
y_all = df['PRES'].values
x_hrd = hrdobs_df['INT'].values
y_hrd = hrdobs_df['PRES'].values

ax2.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.5, zorder=0)

# Plot Scatter
for cat in PLOT_ORDER:
    cat_df = df[df['CAT'] == cat]
    if not cat_df.empty:
        c = COLOR_MAP.get(cat, (0.5, 0.5, 0.5))
        ax2.scatter(cat_df['INT'], cat_df['PRES'], s=SCATTER_SIZE, color=c, 
                    edgecolors=(0.75, 0.75, 0.75), linewidths=MARKER_EDGE_WIDTH, zorder=2)

ax2.scatter(x_hrd, y_hrd, s=SCATTER_SIZE, facecolors='none', edgecolors='black', 
            linewidths=MARKER_EDGE_WIDTH, zorder=3)

# Calculate Regression & R^2
slope_all, int_all = np.polyfit(x_all, y_all, 1)
r2_all = np.corrcoef(x_all, y_all)[0, 1]**2

slope_hrd, int_hrd = np.polyfit(x_hrd, y_hrd, 1)
r2_hrd = np.corrcoef(x_hrd, y_hrd)[0, 1]**2

# Plot Regression Lines
x_fit_all = np.linspace(x_all.min(), x_all.max(), 100)
ax2.plot(x_fit_all, slope_all * x_fit_all + int_all, color=REG_COLOR_ALL, 
         linestyle='-', linewidth=LINE_WIDTH, zorder=4)

x_fit_hrd = np.linspace(x_hrd.min(), x_hrd.max(), 100)
ax2.plot(x_fit_hrd, slope_hrd * x_fit_hrd + int_hrd, color='black', 
         linestyle=REG_STYLE_HRD, linewidth=LINE_WIDTH, zorder=5)

# Formatting
ax2.set_xlim([5, 170])
ax2.set_xticks(range(25, 151, 25))
ax2.set_ylim([890, 1030])
ax2.set_yticks(range(900, 1021, 20))

# Panel B Titles and Labels
ax2.set_title('(b) Wind-Pressure Relationship', fontsize=27, fontname='Arial Narrow', pad=15)
ax2.set_ylabel('MSLP (hPa)', fontsize=20, fontname='Arial Narrow', labelpad=10)
ax2.set_xlabel('Intensity (kt)', fontsize=20, fontname='Arial Narrow', labelpad=10)

# =============================================================================
# 7. EXPORT & TERMINAL OUTPUT
# =============================================================================

output_file = OUTPUT_FIGURE_FILE
plt.savefig(output_file, format='pdf', facecolor='white', edgecolor='none')

print(f"✅ Figure 5 generated successfully: {output_file}")
print("\n=== Data Annotations Summary ===")
print(f"Panel a (TS Cases):")
print(f"  HRDOBS Subset: {total_ts_hrdobs} / {total_hrdobs}")
print(f"  Best Track:    {total_ts_hurdat} / {total_hurdat}")
print(f"\nPanel b (Regression R^2):")
print(f"  Best Track R^2: {r2_all:.3f}")
print(f"  HRDOBS R^2:     {r2_hrd:.3f}")
print("================================\n")
