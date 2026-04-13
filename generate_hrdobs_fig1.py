import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl

# =============================================================================
# 1. I/O DEFINITIONS
# =============================================================================
INPUT_HURDAT_FILE  = 'HURDAT2_all_since_1960_filtered.csv'
OUTPUT_FIGURE_FILE = 'hrdobs_figure1.pdf'

# =============================================================================
# 2. CONFIGURATION & STYLING
# =============================================================================
# Force text to export as editable text boxes instead of paths/individual characters
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# Force all fonts to Arial Narrow
mpl.rcParams['font.sans-serif'] = ['Arial Narrow']
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.linewidth'] = 1.5  # Bold bounding boxes for panels

P1_COLOR, P2_COLOR = 'red', 'green'
DIFF_COLOR = 'black'
LINE_WIDTH_MAIN = 1.5
FIG_W, FIG_H = 12.5, 6.37

# =============================================================================
# 3. DATA LOADING AND ROBUST PREPROCESSING
# =============================================================================
def load_and_process(filename):
    cols = ['ID', 'NAME', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'CAT', 'LAT', 'LON', 'WIND', 'MSLP']
    df_raw = pd.read_csv(filename, names=cols)
    
    # Create Datetime and ensure sorting
    df_raw['DT'] = pd.to_datetime(df_raw[['YEAR', 'MONTH', 'DAY']].assign(hour=df_raw['HOUR'] // 100))
    df_raw = df_raw.sort_values(['ID', 'DT']).drop_duplicates(subset=['ID', 'DT'])

    # Calculate 24-h intensity change per storm ID
    processed_dfs = []
    for storm_id in df_raw['ID'].unique():
        group = df_raw[df_raw['ID'] == storm_id].copy().set_index('DT')
        group['WIND_PRIOR'] = group['WIND'].shift(freq='24h')
        group['INT_CHANGE'] = group['WIND'] - group['WIND_PRIOR']
        processed_dfs.append(group.reset_index())

    return pd.concat(processed_dfs, ignore_index=True)

df = load_and_process(INPUT_HURDAT_FILE)

# =============================================================================
# 4. ANALYTICAL CALCULATIONS
# =============================================================================
years = np.arange(1975, 2025)

# Metrics for timeline panels (Strictly Intensifying and RI)
c_gt0 = np.array([df[df['YEAR'] == y]['INT_CHANGE'].dropna().gt(0).sum() for y in years])
c_ri  = np.array([df[df['YEAR'] == y]['INT_CHANGE'].dropna().ge(30).sum() for y in years])

def get_stats(data):
    """Mean and 95% Confidence Interval."""
    mean = np.mean(data)
    conf = stats.t.ppf(0.975, len(data)-1) * (np.std(data, ddof=1) / np.sqrt(len(data)))
    return mean, conf

# =============================================================================
# 5. PLOTTING
# =============================================================================
fig = plt.figure(figsize=(FIG_W, FIG_H))

def add_panel(x_in, y_in, w_in, h_in):
    return fig.add_axes([x_in / FIG_W, (FIG_H - y_in - h_in) / FIG_H, w_in / FIG_W, h_in / FIG_H])

def plot_stat_lines(ax, x_range, data, color, lw):
    mean, conf = get_stats(data)
    ax.hlines(mean, x_range[0], x_range[1], color=color, linewidth=lw, zorder=5)
    ax.hlines([mean + conf, mean - conf], x_range[0], x_range[1], color=color, 
              linewidth=lw*0.6, linestyle='--', zorder=5)

# --- PANEL A: Histogram (5-kt Bins, Shifted Intervals) ---
ax_a = add_panel(1.02, 0.74, 5, 5)

# Filter for the two periods
h1 = df[(df['YEAR'] >= 1975) & (df['YEAR'] <= 1999)]['INT_CHANGE'].dropna()
h2 = df[(df['YEAR'] >= 2000) & (df['YEAR'] <= 2024)]['INT_CHANGE'].dropna()

bin_edges = np.arange(-115, 125, 5)
cnt1, _ = np.histogram(h1, bins=bin_edges)
cnt2, _ = np.histogram(h2, bins=bin_edges)

bin_centers = bin_edges[:-1] + 2.5
hist_bar_w = 2.0 

# Linewidth set to 0 to remove borders
ax_a.bar(bin_centers - hist_bar_w/2, cnt1, width=hist_bar_w, color=P1_COLOR, alpha=0.8, linewidth=0, label='1975-1999')
ax_a.bar(bin_centers + hist_bar_w/2, cnt2, width=hist_bar_w, color=P2_COLOR, alpha=0.8, linewidth=0, label='2000-2024')

ax_a.set_yscale('log')
ax_a.set_ylim(0.5, 2000)
ax_a.set_xlim(-115, 115) 
ax_a.set_title(r'(a) Intensity Change Spectrum ($x_{i} < x \leq x_{i+1}$)', loc='left', fontweight='bold', fontsize=14)
ax_a.set_xlabel('24-h Intensity Change (kt)', fontsize=12)
ax_a.set_ylabel('Absolute Number of Periods (Log)', fontsize=12)
ax_a.set_xticks(np.arange(-100, 120, 20))
ax_a.grid(True, which="both", axis='y', linestyle=':', alpha=0.3)
ax_a.axvline(0, color='black', linewidth=1.5, linestyle='-')

# Calculate Percent Difference 
pct_diff = np.full_like(cnt1, np.nan, dtype=float)
valid_mask = cnt1 > 0 # Avoid division by zero
pct_diff[valid_mask] = ((cnt2[valid_mask] - cnt1[valid_mask]) / cnt1[valid_mask]) * 100

# Add secondary Y-axis
ax_a_rt = ax_a.twinx()
line_diff, = ax_a_rt.plot(bin_centers, pct_diff, color=DIFF_COLOR, marker='none', markersize=4, 
                          linewidth=1.5, linestyle='-', label='% Difference')

# Draw 0% reference line
ax_a_rt.axhline(0, color=DIFF_COLOR, linestyle='--', linewidth=1, alpha=0.5)

# Format secondary axis
ax_a_rt.set_ylabel('Percent Difference (%)', fontsize=12, color=DIFF_COLOR)
ax_a_rt.tick_params(axis='y', colors=DIFF_COLOR)
ax_a_rt.set_ylim(-100, 300) # Prevents an extreme single outlier from breaking the scale

# Combine legends
handles_a, labels_a = ax_a.get_legend_handles_labels()
handles_a.append(line_diff)
labels_a.append('% Difference (Right Axis)')
ax_a.legend(handles_a, labels_a, frameon=False, loc='upper left')

# --- PANEL B: Strictly Intensifying (> 0 kt) ---
ax_b = add_panel(7.14, 0.74, 5, 2.5)
ax_b.bar(years[:25], c_gt0[:25], color=P1_COLOR, alpha=0.8, width=0.7, linewidth=0)
ax_b.bar(years[25:], c_gt0[25:], color=P2_COLOR, alpha=0.8, width=0.7, linewidth=0)

plot_stat_lines(ax_b, [1974.5, 1999.5], c_gt0[:25], 'black', LINE_WIDTH_MAIN)
plot_stat_lines(ax_b, [1999.5, 2024.5], c_gt0[25:], 'black', LINE_WIDTH_MAIN)

ax_b.set_title(r'(b) Intensifying Storms (> 0 kt / 24 h)', loc='left', fontweight='bold', fontsize=12)
ax_b.set_ylabel('Cycles / Year')
ax_b.set_xlim(1973, 2026)
ax_b.grid(True, linestyle=':', alpha=0.3)

# --- PANEL C: Rapid Intensifications (>= 30 kt) ---
ax_c = add_panel(7.14, 3.24, 5, 2.5)
ax_c.bar(years[:25], c_ri[:25], color=P1_COLOR, alpha=0.8, width=0.7, linewidth=0)
ax_c.bar(years[25:], c_ri[25:], color=P2_COLOR, alpha=0.8, width=0.7, linewidth=0)

plot_stat_lines(ax_c, [1974.5, 1999.5], c_ri[:25], 'black', LINE_WIDTH_MAIN)
plot_stat_lines(ax_c, [1999.5, 2024.5], c_ri[25:], 'black', LINE_WIDTH_MAIN)

ax_c.set_title(r'(c) Rapid Intensifications ($\geq$ 30 kt / 24 h)', loc='left', fontweight='bold', fontsize=12)
ax_c.set_ylabel('Cycles / Year')
ax_c.set_xlabel('Year')
ax_c.set_xlim(1973, 2026)
ax_c.grid(True, linestyle=':', alpha=0.3)

# Save the figure
plt.savefig(OUTPUT_FIGURE_FILE, bbox_inches='tight')
