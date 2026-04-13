import pandas as pd
import graphviz

# =============================================================================
# 1. I/O DEFINITIONS
# =============================================================================
INPUT_HRDOBS_DB_FILE = 'hrdobs_inventory_db.csv'
OUTPUT_FIGURE_FILE   = 'hrdobs_figure3' # Graphviz automatically appends .pdf

def generate_dynamic_schema_figure(db_path=INPUT_HRDOBS_DB_FILE, output_name=OUTPUT_FIGURE_FILE):
    # 1. Read the database to dynamically find all existing groups
    try:
        df = pd.read_csv(db_path)
    except FileNotFoundError:
        print(f"Could not find {db_path}. Please check the path.")
        return

    # Extract all unique observation groups from the CSV
    all_groups = set()
    for obs_str in df['Observation_Groups'].dropna():
        groups = [g.strip() for g in str(obs_str).split(',') if g.strip()]
        all_groups.update(groups)

    # 2. Categorize the groups dynamically
    categories = {
        'dropsonde': {'label': 'dropsonde_{id}<BR/>(Dropsondes)', 'platforms': []},
        'flight_level': {'label': 'flight_level_{id}<BR/>(In-Situ)', 'platforms': []},
        'sfmr': {'label': 'sfmr_{id}<BR/>(Radiometer)', 'platforms': []},
        'tdr': {'label': 'tdr_{id}<BR/>(Radar)', 'platforms': []},
        'track': {'label': 'track_{type}<BR/>(Storm Tracks)', 'platforms': []},
        'ships': {'label': 'ships_params<BR/>(Environment)', 'platforms': []}
    }

    for g in all_groups:
        if g.startswith('dropsonde_'):
            categories['dropsonde']['platforms'].append(g.replace('dropsonde_', ''))
        elif g.startswith('flight_level_hdobs_'):
            categories['flight_level']['platforms'].append(g.replace('flight_level_hdobs_', ''))
        elif g.startswith('sfmr_'):
            categories['sfmr']['platforms'].append(g.replace('sfmr_', ''))
        elif g.startswith('tdr_'):
            categories['tdr']['platforms'].append(g.replace('tdr_', ''))
        elif g.startswith('track_'):
            categories['track']['platforms'].append(g.replace('track_', ''))
        elif g == 'ships_params':
            categories['ships']['platforms'].append('SHIPS Predictors')

    for cat in categories.values():
        cat['platforms'] = sorted(cat['platforms'])

    # 3. Build the Graphviz Flowchart
    dot = graphviz.Digraph(comment='HRDOBS Schema', format='pdf')
    
    # Layout settings
    dot.attr(rankdir='TB', size='11,8.5', fontname='Arial, Helvetica, sans-serif', nodesep='0.2', ranksep='0.6')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='#f8f9fa', 
             fontname='Arial, Helvetica, sans-serif', fontsize='12', color='#343a40', penwidth='1.5')
    dot.attr('edge', color='#495057', penwidth='1.5', arrowsize='0.8')

    # --- ROW 1: Root Node ---
    root_label = "<<B>INPUT HDF5 FILE</B>>"
    dot.node('root', root_label, fillcolor='#e9ecef', fontsize='14')

    # --- ROW 2: Metadata Node (White, Centered, No Bullets) ---
    meta_label = (
        "<<B>GLOBAL METADATA</B><BR/>"
        "storm_name / storm_id<BR/>"
        "storm_motion (vector)<BR/>"
        "geospatial_bounds<BR/>"
        "platforms<BR/>"
        "storm_center (NHC)>"
    )
    dot.node('meta', meta_label, shape='note', fillcolor='#ffffff')

    # --- ROW 2: Observation Group Nodes ---
    active_groups = []
    for key, cat in categories.items():
        if not cat['platforms']:
            continue
        active_groups.append(key)
        
        # Center aligned text with no bullets
        plat_str = "<BR/>".join(cat['platforms'])
        node_label = f"<<B>{cat['label']}</B><BR/><BR/>Platforms/Sources:<BR/>{plat_str}>"
        
        # Apply specific colors: Blue for instruments, Green for tracks & SHIPS
        if key in ['dropsonde', 'flight_level', 'sfmr', 'tdr']:
            color = '#cce5ff'  # Clean, readable blue
        else:
            color = '#d4edda'  # Clean, readable green
            
        dot.node(key, node_label, fillcolor=color)

    # Invisible Center Spine Node for Row 2
    dot.node('center2', '', shape='point', style='invis', width='0', height='0')

    # --- ROW 3: Standardized Arrays Node (Centered, No Bullets) ---
    arrays_label = (
        "<<B>STANDARDIZED ARRAYS</B><BR/>"
        "1D Data (p, t, w, rr, etc.)<BR/>"
        "QC / error estimates<BR/>"
        "Unified Units (SI)<BR/>"
        "Cleaned NaNs>"
    )
    dot.node('arrays', arrays_label, fillcolor='#e9ecef', shape='cylinder')

    # --- ROW 4: Final AI-Ready HDF5 File Node ---
    final_label = "<<B>FINAL AI-READY HDF5 FILE</B>>"
    dot.node('final', final_label, fillcolor='#d6d8db', shape='folder', fontsize='14')
    
    # --- ROW 5: Applications Node (Centered, No Bullets) ---
    apps_label = (
        "<<B>DOWNSTREAM APPLICATIONS</B><BR/>"
        "AI / ML / DA Applications<BR/>"
        "Physics / Dynamics<BR/>"
        "Instrument Cal-Val>"
    )
    dot.node('apps', apps_label, fillcolor='#e2e3e5', style='filled,rounded')

    # =========================================================================
    # --- SYMMETRICAL CENTERING LOGIC ---
    # =========================================================================
    
    # 1. Establish the main vertical spine with extremely high weight
    dot.edge('root', 'center2', style='invis', weight='1000')
    dot.edge('center2', 'arrays', style='invis', weight='1000')
    dot.edge('arrays', 'final', weight='1000')
    dot.edge('final', 'apps', weight='1000')
    
    # 2. Add Row 2 Nodes to the same rank and distribute them left-to-right
    with dot.subgraph() as s:
        s.attr(rank='same')
        row2_nodes = ['meta'] + active_groups
        
        # Find the physical median to insert the invisible center node
        mid_idx = len(row2_nodes) // 2
        row2_nodes.insert(mid_idx, 'center2')
        
        # String them together horizontally (light weight so it doesn't skew the spine)
        for i in range(len(row2_nodes)-1):
            s.edge(row2_nodes[i], row2_nodes[i+1], style='invis', weight='2')

    # 3. Connect the Root down to the Row 2 components
    dot.edge('root', 'meta')
    for key in active_groups:
        dot.edge('root', key)

    # 4. Connect the Row 2 Observation Groups down to Arrays
    for key in active_groups:
        dot.edge(key, 'arrays')

    # 5. Draw the 90-degree dashed bracket around the left side
    dot.edge('meta:w', 'final:w', style='dashed', color='#0056b3')

    # 4. Render the PDF
    try:
        dot.render(output_name, cleanup=True)
        print(f"Successfully generated finalized schema figure: {output_name}.pdf")
    except Exception as e:
        print(f"Error rendering PDF. Ensure Graphviz is installed on your system. Details: {e}")

if __name__ == "__main__":
    generate_dynamic_schema_figure()
    