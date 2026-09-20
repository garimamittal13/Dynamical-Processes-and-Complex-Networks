import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ==========================================
# 1. DATA INGESTION & CLEANING
# ==========================================
df = pd.read_csv('Survey_Results_UC.csv', encoding='utf-8-sig', sep=None, engine='python')

# Safely drop ONLY the actual ID column
id_cols = [col for col in df.columns if col.lower().startswith('id')]
df_responses = df.drop(columns=id_cols)

# Map Likert scale to numerical values
mapping = {
    'Strongly Disagree': 1, 'Disagree': 2, 'Neutral': 3,
    'Agree': 4, 'Strongly Agree': 5, 'No Comments': 3
}

# Clean whitespace, map strings to numbers, and coerce errors to NaN
df_responses = df_responses.replace(r'^\s+|\s+$', '', regex=True)
df_responses = df_responses.replace(mapping)
df_responses = df_responses.apply(pd.to_numeric, errors='coerce')

# Drop rows where ALL answers are completely empty
df_responses = df_responses.dropna(how='all')
df_responses = df_responses.fillna(3)

# Rename columns to their 3-character codes (T01, E12, etc.)
rename_dict = {col: col[:3] for col in df_responses.columns}
df_responses = df_responses.rename(columns=rename_dict)

# Calculate the Pearson correlation matrix
corr = df_responses.corr()


# ==========================================
# 2. CALCULATING CENTRALITY METRICS
# ==========================================
PRIMARY_THRESHOLD = 0.33 
G = nx.Graph()
cols = corr.columns

for i in range(len(cols)):
    node_name = cols[i]
    G.add_node(node_name, category=node_name[0])
    for j in range(i+1, len(cols)):
        weight = corr.iloc[i, j]
        if abs(weight) > PRIMARY_THRESHOLD:
            G.add_edge(node_name, cols[j], weight=abs(weight))

degree_cent = nx.degree_centrality(G)
between_cent = nx.betweenness_centrality(G)
eigen_cent = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)

print(f"--- NETWORK METRICS (Calculated at Threshold = {PRIMARY_THRESHOLD}) ---")
print(f"Total Nodes: {G.number_of_nodes()} | Total Edges: {G.number_of_edges()}")

categories = ['T', 'E', 'S', 'V']
cat_names = {'T': 'Technology', 'E': 'Education', 'S': 'Ethics', 'V': 'Environment'}

# Create a set to store the top predictor nodes for bolding
important_nodes = set()

for cat in categories:
    cat_nodes = [n for n, d in G.nodes(data=True) if d['category'] == cat]
    
    cat_deg = sorted({n: degree_cent[n] for n in cat_nodes}.items(), key=lambda x: x[1], reverse=True)[:4]
    cat_bet = sorted({n: between_cent[n] for n in cat_nodes}.items(), key=lambda x: x[1], reverse=True)[:4]
    cat_eig = sorted({n: eigen_cent[n] for n in cat_nodes}.items(), key=lambda x: x[1], reverse=True)[:4]
    
    # Add the top degree nodes (predictors) to our important_nodes set
    important_nodes.update([n for n, v in cat_deg])
    
    print(f"\n[{cat_names[cat]} - {cat}]")
    print(f"  Top Degree (Predictors): {[n for n, v in cat_deg]}")
    print(f"  Top Betweenness (Bridges): {[n for n, v in cat_bet]}")
    print(f"  Top Eigenvector (Influencers): {[n for n, v in cat_eig]}")


# ==========================================
# 3. ADVANCED SPREAD VISUALIZATION
# ==========================================
fig, ax = plt.subplots(figsize=(22, 14))

main_nodes = [n for n in G.nodes if G.degree(n) > 0]
isolated_nodes = [n for n in G.nodes if G.degree(n) == 0]

# Layout for the main network: low iterations prevent the tight clustering
pos = nx.spring_layout(G.subgraph(main_nodes), k=0.8, iterations=30, seed=42)

# Normalize and manually scale the main network positions to force them apart
max_val = max(max(abs(x), abs(y)) for x, y in pos.values())
if max_val > 0:
    for n in pos:
        pos[n] = (pos[n] / max_val) * 1.5

# Arrange the isolated (disconnected) nodes in a perfect circle around the main network
if isolated_nodes:
    radius = 1.9  # Placed in an orbit outside the normalized main network
    angles = np.linspace(0, 2 * np.pi, len(isolated_nodes), endpoint=False)
    for i, node in enumerate(isolated_nodes):
        pos[node] = np.array([radius * np.cos(angles[i]), radius * np.sin(angles[i])])

color_map = {'T': '#ADD8E6', 'E': '#90EE90', 'S': '#F08080', 'V': '#FFD700'}
colors = [color_map[G.nodes[n]['category']] for n in G.nodes]

# Set line widths: bold (3.0) for predictors, normal (1.2) for the rest
line_widths = [3.0 if n in important_nodes else 1.2 for n in G.nodes]

# Draw edges lighter to reduce visual congestion
nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, edge_color='gray', width=0.8)

# Draw nodes slightly smaller to prevent overlap, applying the dynamic line_widths
nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=600, edgecolors='black', linewidths=line_widths)

# Draw labels
nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight='bold')

# --- Top Right Legend & Metrics Setup ---
legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#ADD8E6', markeredgecolor='black', markersize=8, label='Technology'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#90EE90', markeredgecolor='black', markersize=8, label='Education'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#F08080', markeredgecolor='black', markersize=8, label='Social & Ethics'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD700', markeredgecolor='black', markersize=8, label='Environment'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor='black', markeredgewidth=3, markersize=8, label='Bold = Top Predictor')
]

# Place legend in top right corner
ax.legend(handles=legend_handles, loc='upper right', fontsize=8, frameon=True, shadow=True)

# Anchor the metrics text perfectly beneath the legend, adding the threshold
metrics_text = (f"Connected: {len(main_nodes)} | Isolated: {len(isolated_nodes)} | Edges: {G.number_of_edges()}\n"
                f"Correlation Threshold: > {PRIMARY_THRESHOLD}")
ax.text(1, 0.83, metrics_text, transform=ax.transAxes, fontsize=8, 
        horizontalalignment='right', verticalalignment='top', fontweight='bold')
# ----------------------------------------

# Broaden canvas limits to fit the circular orbit without clipping
ax.set_xlim(-2.1, 2.1)
ax.set_ylim(-2.1, 2.1)
ax.axis('off')

plt.tight_layout()
plt.show()