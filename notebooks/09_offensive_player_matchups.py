# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: venv (3.12.10)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # NFL Big Data Bowl 2026 - Offensive Player Matchup Matrix

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['font.size'] = 11

print("✅ Setup complete!")

# %% [markdown]
# ## 1. Load Data

# %%
DATA_DIR = Path(r'../data')
PROCESSED_DIR = DATA_DIR / 'processed'
COMPETITION_DIR = DATA_DIR / '114239_nfl_competition_files_published_analytics_final'
TRAIN_DIR = COMPETITION_DIR / 'train'
OUTPUT_DIR = PROCESSED_DIR / 'offensive_matchups'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading data...")

convergence_df = pd.read_csv(PROCESSED_DIR / 'convergence_speed_all_plays.csv')
print(f"✓ Convergence data: {len(convergence_df):,} rows")

supp_data = pd.read_csv(COMPETITION_DIR / 'supplementary_data.csv')
print(f"✓ Supplementary data: {len(supp_data):,} rows")

merged_df = convergence_df.merge(
    supp_data,
    on=['game_id', 'play_id', 'week'],
    how='left'
)

print(f"✓ Merged data: {len(merged_df):,} rows")

# Filter for OFFENSIVE players (targeted receivers)
offense = merged_df[merged_df['player_role'] == 'Targeted Receiver'].copy()
print(f"✓ Offensive player (receiver) instances: {len(offense):,}")

# %%
# LOAD PLAYER NAMES FROM ALL INPUT FILES
print("\n" + "="*80)
print("LOADING PLAYER NAMES FROM ALL INPUT FILES")
print("="*80)

input_files = sorted(TRAIN_DIR.glob('input_2023_w*.csv'))
print(f"Found {len(input_files)} input files")

all_players = []

if len(input_files) > 0:
    for input_file in input_files:
        print(f"Loading: {input_file.name}...", end=" ")
        try:
            input_df = pd.read_csv(input_file)
            
            if 'nfl_id' in input_df.columns and 'player_name' in input_df.columns:
                file_players = input_df[['nfl_id', 'player_name', 'player_position']].drop_duplicates('nfl_id')
                all_players.append(file_players)
                print(f"✓ {len(file_players)} unique players")
            else:
                print(f"⚠️ Missing required columns")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    if len(all_players) > 0:
        player_lookup = pd.concat(all_players, ignore_index=True).drop_duplicates('nfl_id', keep='first')
        player_lookup.columns = ['nfl_id', 'player_name', 'position']
        
        print(f"\n✅ TOTAL UNIQUE PLAYERS LOADED: {len(player_lookup)}")
        
        # Add team info for OFFENSE
        if 'offensive_team' in offense.columns:
            team_lookup = offense.groupby('nfl_id')['offensive_team'].agg(
                lambda x: x.mode()[0] if len(x.mode()) > 0 else 'UNK'
            ).reset_index()
            team_lookup.columns = ['nfl_id', 'team']
            player_lookup = player_lookup.merge(team_lookup, on='nfl_id', how='left')
            player_lookup['team'] = player_lookup['team'].fillna('UNK')
            print(f"✓ Added team info")
        
        # Filter offense to only include players we have names for
        print(f"\nBefore filtering: {offense['nfl_id'].nunique()} unique offensive players")
        offense = offense[offense['nfl_id'].isin(player_lookup['nfl_id'])].copy()
        print(f"After filtering: {offense['nfl_id'].nunique()} unique offensive players")
        print(f"✓ Filtered to only players with metadata\n")
        
        print(f"📋 Sample of loaded player names:")
        print(player_lookup.head(15).to_string(index=False))
    else:
        print("❌ No player data could be loaded!")
        player_lookup = None
else:
    print("❌ No input files found!")
    player_lookup = None

# %% [markdown]
# ## 2. Receiver Performance by Coverage Type

# %%
print("="*80)
print("RECEIVER × COVERAGE TYPE PERFORMANCE ANALYSIS")
print("="*80)
print("Higher convergence = Better separation (GOOD for offense)")
print("="*80)

MIN_PLAYS_COV = 5

receiver_coverage_perf = offense.groupby(['nfl_id', 'team_coverage_type']).agg({
    'convergence_speed': ['mean', 'std', 'count'],
    'min_distance': 'mean',
    'player_position': lambda x: x.mode()[0] if len(x) > 0 else None
}).round(3)

receiver_coverage_perf.columns = ['Avg Conv', 'Std Dev', 'N Plays', 'Avg Min Dist', 'Position']
receiver_coverage_perf = receiver_coverage_perf[receiver_coverage_perf['N Plays'] >= MIN_PLAYS_COV].reset_index()

print(f"\nReceivers with coverage-specific data: {receiver_coverage_perf['nfl_id'].nunique()}")
print(f"Total receiver-coverage combinations: {len(receiver_coverage_perf)}")

# Identify coverage strengths/weaknesses
print("\n" + "="*80)
print("TOP RECEIVERS WHO DOMINATE/STRUGGLE VS SPECIFIC COVERAGES")
print("="*80)

coverage_matchups = []

for player_id in receiver_coverage_perf['nfl_id'].unique():
    player_data = receiver_coverage_perf[receiver_coverage_perf['nfl_id'] == player_id]
    
    if len(player_data) < 2:
        continue
    
    # For offense, HIGHEST convergence = best (getting separation)
    best_cov_idx = player_data['Avg Conv'].idxmax()
    best_cov = player_data.loc[best_cov_idx]
    
    worst_cov_idx = player_data['Avg Conv'].idxmin()
    worst_cov = player_data.loc[worst_cov_idx]
    
    coverage_gap = best_cov['Avg Conv'] - worst_cov['Avg Conv']
    
    coverage_matchups.append({
        'nfl_id': player_id,
        'best_coverage': best_cov['team_coverage_type'],
        'best_coverage_conv': best_cov['Avg Conv'],
        'best_coverage_plays': best_cov['N Plays'],
        'worst_coverage': worst_cov['team_coverage_type'],
        'worst_coverage_conv': worst_cov['Avg Conv'],
        'worst_coverage_plays': worst_cov['N Plays'],
        'coverage_gap': coverage_gap
    })

cov_matchup_df = pd.DataFrame(coverage_matchups)
cov_matchup_df = cov_matchup_df.sort_values('coverage_gap', ascending=False)

# Add player names
if player_lookup is not None:
    cov_matchup_df = cov_matchup_df.merge(
        player_lookup[['nfl_id', 'player_name', 'team', 'position']], 
        on='nfl_id', 
        how='left'
    )
    cov_matchup_df['player_name'] = cov_matchup_df['player_name'].fillna('Unknown Player')
    cov_matchup_df['team'] = cov_matchup_df['team'].fillna('UNK')
    cov_matchup_df['position'] = cov_matchup_df['position'].fillna('UNK')
    
    cols = ['nfl_id', 'player_name', 'team', 'position', 'best_coverage', 'best_coverage_conv',
            'best_coverage_plays', 'worst_coverage', 'worst_coverage_conv', 'worst_coverage_plays', 'coverage_gap']
    cov_matchup_df = cov_matchup_df[cols]

print("\nTop 20 Receivers with Biggest Coverage Performance Gaps:")
print("-" * 80)
if player_lookup is not None and 'player_name' in cov_matchup_df.columns:
    display_cols = ['player_name', 'team', 'position', 'best_coverage', 'best_coverage_conv',
                    'worst_coverage', 'worst_coverage_conv', 'coverage_gap']
    print(cov_matchup_df[display_cols].head(20).to_string(index=False))
else:
    print(cov_matchup_df.head(20).to_string(index=False))

cov_matchup_df.to_csv(OUTPUT_DIR / 'receiver_coverage_matchups.csv', index=False)

# %% [markdown]
# ## 3. Receiver Performance by Route Type

# %%
print("\n" + "="*80)
print("RECEIVER × ROUTE TYPE PERFORMANCE ANALYSIS")
print("="*80)

MIN_PLAYS_ROUTE = 3

receiver_route_perf = offense.groupby(['nfl_id', 'route_of_targeted_receiver']).agg({
    'convergence_speed': ['mean', 'std', 'count'],
    'min_distance': 'mean',
    'player_position': lambda x: x.mode()[0] if len(x) > 0 else None
}).round(3)

receiver_route_perf.columns = ['Avg Conv', 'Std Dev', 'N Plays', 'Avg Min Dist', 'Position']
receiver_route_perf = receiver_route_perf[receiver_route_perf['N Plays'] >= MIN_PLAYS_ROUTE].reset_index()

print(f"\nReceivers with route-specific data: {receiver_route_perf['nfl_id'].nunique()}")

route_strengths = []

for player_id in receiver_route_perf['nfl_id'].unique():
    player_data = receiver_route_perf[receiver_route_perf['nfl_id'] == player_id]
    
    if len(player_data) < 2:
        continue
    
    best_route_idx = player_data['Avg Conv'].idxmax()
    best_route = player_data.loc[best_route_idx]
    
    worst_route_idx = player_data['Avg Conv'].idxmin()
    worst_route = player_data.loc[worst_route_idx]
    
    route_gap = best_route['Avg Conv'] - worst_route['Avg Conv']
    
    route_strengths.append({
        'nfl_id': player_id,
        'best_route': best_route['route_of_targeted_receiver'],
        'best_route_conv': best_route['Avg Conv'],
        'best_route_plays': best_route['N Plays'],
        'worst_route': worst_route['route_of_targeted_receiver'],
        'worst_route_conv': worst_route['Avg Conv'],
        'worst_route_plays': worst_route['N Plays'],
        'route_gap': route_gap
    })

route_strength_df = pd.DataFrame(route_strengths)
route_strength_df = route_strength_df.sort_values('route_gap', ascending=False)

if player_lookup is not None:
    route_strength_df = route_strength_df.merge(
        player_lookup[['nfl_id', 'player_name', 'team', 'position']], 
        on='nfl_id', 
        how='left'
    )
    route_strength_df['player_name'] = route_strength_df['player_name'].fillna('Unknown Player')
    route_strength_df['team'] = route_strength_df['team'].fillna('UNK')
    route_strength_df['position'] = route_strength_df['position'].fillna('UNK')
    
    cols = ['nfl_id', 'player_name', 'team', 'position', 'best_route', 'best_route_conv',
            'best_route_plays', 'worst_route', 'worst_route_conv', 'worst_route_plays', 'route_gap']
    route_strength_df = route_strength_df[cols]

print("\nTop 20 Receivers with Route-Specific Strengths:")
print("-" * 80)
if player_lookup is not None and 'player_name' in route_strength_df.columns:
    display_cols = ['player_name', 'team', 'position', 'best_route', 'best_route_conv',
                    'worst_route', 'worst_route_conv', 'route_gap']
    print(route_strength_df[display_cols].head(20).to_string(index=False))
else:
    print(route_strength_df.head(20).to_string(index=False))

route_strength_df.to_csv(OUTPUT_DIR / 'receiver_route_strengths.csv', index=False)

# %% [markdown]
# ## 4. Receiver vs Defender Position Performance

# %%
print("\n" + "="*80)
print("RECEIVER × DEFENDER POSITION MATCHUP ANALYSIS")
print("="*80)

MIN_PLAYS_DEF = 5

# First, we need to get the defender position from the merged data
# This requires matching defenders for each play
receiver_vs_def = offense.groupby(['nfl_id', 'defender_position']).agg({
    'convergence_speed': ['mean', 'count'],
    'min_distance': 'mean',
    'player_position': lambda x: x.mode()[0] if len(x) > 0 else None
}).round(3)

receiver_vs_def.columns = ['Avg Conv', 'N Plays', 'Avg Min Dist', 'Position']
receiver_vs_def = receiver_vs_def[receiver_vs_def['N Plays'] >= MIN_PLAYS_DEF].reset_index()

print(f"\nReceivers with defender-specific data: {receiver_vs_def['nfl_id'].nunique()}")

def_matchups = []

for player_id in receiver_vs_def['nfl_id'].unique():
    player_data = receiver_vs_def[receiver_vs_def['nfl_id'] == player_id]
    
    if len(player_data) < 2:
        continue
    
    best_def_idx = player_data['Avg Conv'].idxmax()
    best_def = player_data.loc[best_def_idx]
    
    worst_def_idx = player_data['Avg Conv'].idxmin()
    worst_def = player_data.loc[worst_def_idx]
    
    def_gap = best_def['Avg Conv'] - worst_def['Avg Conv']
    
    def_matchups.append({
        'nfl_id': player_id,
        'beats_position': best_def['defender_position'],
        'beats_conv': best_def['Avg Conv'],
        'beats_plays': best_def['N Plays'],
        'struggles_vs': worst_def['defender_position'],
        'struggles_conv': worst_def['Avg Conv'],
        'struggles_plays': worst_def['N Plays'],
        'position_gap': def_gap
    })

def_matchup_df = pd.DataFrame(def_matchups)
def_matchup_df = def_matchup_df.sort_values('position_gap', ascending=False)

if player_lookup is not None:
    def_matchup_df = def_matchup_df.merge(
        player_lookup[['nfl_id', 'player_name', 'team', 'position']], 
        on='nfl_id', 
        how='left'
    )
    def_matchup_df['player_name'] = def_matchup_df['player_name'].fillna('Unknown Player')
    def_matchup_df['team'] = def_matchup_df['team'].fillna('UNK')
    def_matchup_df['position'] = def_matchup_df['position'].fillna('UNK')
    
    cols = ['nfl_id', 'player_name', 'team', 'position', 'beats_position', 'beats_conv',
            'beats_plays', 'struggles_vs', 'struggles_conv', 'struggles_plays', 'position_gap']
    def_matchup_df = def_matchup_df[cols]

print("\nTop 20 Receivers with Position-Specific Matchup Advantages:")
print("-" * 80)
if player_lookup is not None and 'player_name' in def_matchup_df.columns:
    display_cols = ['player_name', 'team', 'position', 'beats_position', 'beats_conv',
                    'struggles_vs', 'struggles_conv', 'position_gap']
    print(def_matchup_df[display_cols].head(20).to_string(index=False))
else:
    print(def_matchup_df.head(20).to_string(index=False))

def_matchup_df.to_csv(OUTPUT_DIR / 'receiver_vs_defender_position.csv', index=False)

# %% [markdown]
# ## 5. Receiver Performance by Distance

# %%
print("\n" + "="*80)
print("RECEIVER × ROUTE DEPTH ANALYSIS")
print("="*80)

offense['distance_category'] = pd.cut(
    offense['initial_distance'],
    bins=[0, 10, 15, 20, 100],
    labels=['Short (<10y)', 'Medium (10-15y)', 'Deep (15-20y)', 'Bomb (>20y)']
)

MIN_PLAYS_DIST = 5

receiver_distance_perf = offense.groupby(['nfl_id', 'distance_category']).agg({
    'convergence_speed': ['mean', 'count'],
    'min_distance': 'mean',
    'player_position': lambda x: x.mode()[0] if len(x) > 0 else None
}).round(3)

receiver_distance_perf.columns = ['Avg Conv', 'N Plays', 'Avg Min Dist', 'Position']
receiver_distance_perf = receiver_distance_perf[receiver_distance_perf['N Plays'] >= MIN_PLAYS_DIST].reset_index()

print(f"\nReceivers with distance-specific data: {receiver_distance_perf['nfl_id'].nunique()}")

distance_strengths = []

for player_id in receiver_distance_perf['nfl_id'].unique():
    player_data = receiver_distance_perf[receiver_distance_perf['nfl_id'] == player_id]
    
    if len(player_data) < 2:
        continue
    
    best_dist_idx = player_data['Avg Conv'].idxmax()
    best_dist = player_data.loc[best_dist_idx]
    
    worst_dist_idx = player_data['Avg Conv'].idxmin()
    worst_dist = player_data.loc[worst_dist_idx]
    
    dist_gap = best_dist['Avg Conv'] - worst_dist['Avg Conv']
    
    distance_strengths.append({
        'nfl_id': player_id,
        'best_depth': best_dist['distance_category'],
        'best_depth_conv': best_dist['Avg Conv'],
        'best_depth_plays': best_dist['N Plays'],
        'worst_depth': worst_dist['distance_category'],
        'worst_depth_conv': worst_dist['Avg Conv'],
        'worst_depth_plays': worst_dist['N Plays'],
        'depth_gap': dist_gap
    })

dist_strength_df = pd.DataFrame(distance_strengths)
dist_strength_df = dist_strength_df.sort_values('depth_gap', ascending=False)

if player_lookup is not None:
    dist_strength_df = dist_strength_df.merge(
        player_lookup[['nfl_id', 'player_name', 'team', 'position']], 
        on='nfl_id', 
        how='left'
    )
    dist_strength_df['player_name'] = dist_strength_df['player_name'].fillna('Unknown Player')
    dist_strength_df['team'] = dist_strength_df['team'].fillna('UNK')
    dist_strength_df['position'] = dist_strength_df['position'].fillna('UNK')
    
    cols = ['nfl_id', 'player_name', 'team', 'position', 'best_depth', 'best_depth_conv',
            'best_depth_plays', 'worst_depth', 'worst_depth_conv', 'worst_depth_plays', 'depth_gap']
    dist_strength_df = dist_strength_df[cols]

print("\nTop 20 Receivers with Depth-Specific Strengths:")
print("-" * 80)
if player_lookup is not None and 'player_name' in dist_strength_df.columns:
    display_cols = ['player_name', 'team', 'position', 'best_depth', 'best_depth_conv',
                    'worst_depth', 'worst_depth_conv', 'depth_gap']
    print(dist_strength_df[display_cols].head(20).to_string(index=False))
else:
    print(dist_strength_df.head(20).to_string(index=False))

dist_strength_df.to_csv(OUTPUT_DIR / 'receiver_depth_strengths.csv', index=False)

# %% [markdown]
# ## 6. Comprehensive Receiver Scouting Cards

# %%
def generate_receiver_scouting_card(player_id, data, player_lookup=None):
    """Generate comprehensive receiver scouting report"""
    player_data = data[data['nfl_id'] == player_id]
    
    if len(player_data) == 0:
        return None
    
    card = {
        'nfl_id': player_id,
        'position': player_data['player_position'].mode()[0] if len(player_data) > 0 else 'UNK',
        'total_plays': len(player_data),
        'avg_convergence': player_data['convergence_speed'].mean(),
        'avg_min_distance': player_data['min_distance'].mean(),
        'avg_speed': player_data['avg_speed'].mean()
    }
    
    if player_lookup is not None:
        player_info = player_lookup[player_lookup['nfl_id'] == player_id]
        if len(player_info) > 0:
            card['player_name'] = player_info['player_name'].iloc[0]
            if 'team' in player_info.columns:
                card['team'] = player_info['team'].iloc[0]
    
    # Best/worst routes
    route_perf = player_data.groupby('route_of_targeted_receiver').agg({
        'convergence_speed': 'mean',
        'game_id': 'count'
    }).round(3)
    route_perf.columns = ['Avg Conv', 'N']
    route_perf = route_perf[route_perf['N'] >= 3].sort_values('Avg Conv', ascending=False)
    
    card['best_routes'] = route_perf.head(3).to_dict()
    card['worst_routes'] = route_perf.tail(3).to_dict()
    
    # Coverage performance
    if 'team_coverage_type' in player_data.columns:
        cov_perf = player_data.groupby('team_coverage_type').agg({
            'convergence_speed': 'mean',
            'game_id': 'count'
        }).round(3)
        cov_perf.columns = ['Avg Conv', 'N']
        cov_perf = cov_perf[cov_perf['N'] >= 3].sort_values('Avg Conv', ascending=False)
        
        card['coverage_performance'] = cov_perf.to_dict()
    
    # Distance performance
    player_data_dist = player_data.copy()
    player_data_dist['dist_cat'] = pd.cut(
        player_data_dist['initial_distance'],
        bins=[0, 10, 15, 20, 100],
        labels=['Short', 'Medium', 'Deep', 'Bomb']
    )
    
    dist_perf = player_data_dist.groupby('dist_cat').agg({
        'convergence_speed': 'mean',
        'game_id': 'count'
    }).round(3)
    dist_perf.columns = ['Avg Conv', 'N']
    dist_perf = dist_perf[dist_perf['N'] >= 3]
    
    card['distance_performance'] = dist_perf.to_dict()
    
    return card

print("\n" + "="*80)
print("GENERATING INDIVIDUAL RECEIVER SCOUTING CARDS")
print("="*80)

top_cov_players = cov_matchup_df.head(10)['nfl_id'].tolist()
top_route_players = route_strength_df.head(10)['nfl_id'].tolist()
top_dist_players = dist_strength_df.head(10)['nfl_id'].tolist()

priority_receivers = list(set(top_cov_players + top_route_players + top_dist_players))

receiver_cards = {}

for player_id in priority_receivers:
    card = generate_receiver_scouting_card(player_id, offense, player_lookup)
    if card:
        receiver_cards[player_id] = card
        name_str = f"{card.get('player_name', f'Player #{player_id}')}"
        team_str = f" ({card.get('team', 'UNK')})" if 'team' in card else ""
        pos_str = f" - {card['position']}"
        print(f"✓ Generated card for {name_str}{team_str}{pos_str}")

print(f"\n✅ Generated {len(receiver_cards)} receiver scouting cards")

# %% [markdown]
# ## 7. Visualize Receiver Matchup Matrices

# %%
print("\n" + "="*80)
print("CREATING RECEIVER-COVERAGE HEATMAP")
print("="*80)

top_receivers = cov_matchup_df.head(20)['nfl_id'].tolist()

receiver_cov_matrix = receiver_coverage_perf[receiver_coverage_perf['nfl_id'].isin(top_receivers)]

if player_lookup is not None:
    receiver_cov_matrix = receiver_cov_matrix.merge(
        player_lookup[['nfl_id', 'player_name', 'team']], 
        on='nfl_id', 
        how='left'
    )
    receiver_cov_matrix['player_name'] = receiver_cov_matrix['player_name'].fillna('Unknown')
    receiver_cov_matrix['team'] = receiver_cov_matrix['team'].fillna('UNK')
    
    receiver_cov_matrix['player_label'] = receiver_cov_matrix.apply(
        lambda x: f"{x['player_name']} ({x['team']})", 
        axis=1
    )
    index_col = 'player_label'
else:
    receiver_cov_matrix['player_label'] = receiver_cov_matrix['nfl_id'].apply(lambda x: f"Player #{int(x)}")
    index_col = 'player_label'

matrix_pivot = receiver_cov_matrix.pivot_table(
    values='Avg Conv',
    index=index_col,
    columns='team_coverage_type',
    aggfunc='mean'
)

if len(matrix_pivot) > 0:
    matrix_pivot.to_csv(OUTPUT_DIR / 'receiver_coverage_matrix.csv')
    
    print(f"\n✅ Receiver-Coverage Matrix Created ({len(matrix_pivot)} receivers × {len(matrix_pivot.columns)} coverages)")
    print("\nTop 10 receivers in matrix:")
    print(matrix_pivot.head(10))
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    sns.heatmap(matrix_pivot, annot=True, fmt='.2f', cmap='RdYlGn', 
               center=0, ax=ax, cbar_kws={'label': 'Avg Convergence (yd/s)'})
    
    ax.set_title('Receiver-Coverage Performance Matrix (Top 20 Receivers)\n' + 
                'Higher values (green) = Better separation/performance',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Coverage Type', fontsize=12)
    ax.set_ylabel('Receiver Name (Team)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'receiver_coverage_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Heatmap saved!")
else:
    print("⚠️ No data for heatmap")

# %% [markdown]
# ## 8. Receiver Visual Cards

# %%
def create_visual_receiver_card(player_id, card, filename):
    """Create visual quick reference card for a receiver"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    player_name = card.get('player_name', f'Player #{player_id}')
    team = card.get('team', 'UNK')
    position = card['position']
    
    title = f'{player_name} ({team}) - {position}\nOFFENSIVE SCOUTING CARD'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Overall stats
    ax = axes[0, 0]
    ax.axis('off')
    
    stats_text = f"""
OVERALL STATS
━━━━━━━━━━━━━━━━━━
Total Targets: {card['total_plays']}
Avg Separation: {card['avg_convergence']:.3f} yd/s
Avg Min Distance: {card['avg_min_distance']:.2f} yards
Avg Speed: {card['avg_speed']:.2f} yd/s
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Best routes
    ax = axes[0, 1]
    ax.axis('off')
    
    routes_text = "⭐ BEST ROUTES\n" + "━"*25 + "\n"
    if 'best_routes' in card and 'Avg Conv' in card['best_routes']:
        for route, conv in list(card['best_routes']['Avg Conv'].items())[:3]:
            n = card['best_routes']['N'].get(route, 0)
            routes_text += f"• {route}: {conv:.3f} yd/s ({int(n)})\n"
    
    ax.text(0.1, 0.5, routes_text, fontsize=10, family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Coverage matchups
    ax = axes[1, 0]
    ax.axis('off')
    
    cov_text = "🎯 BEST VS COVERAGE\n" + "━"*25 + "\n"
    if 'coverage_performance' in card and 'Avg Conv' in card['coverage_performance']:
        for cov, conv in list(card['coverage_performance']['Avg Conv'].items())[:3]:
            cov_text += f"• {cov}: {conv:.3f} yd/s\n"
    
    ax.text(0.1, 0.5, cov_text, fontsize=10, family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Distance profile
    ax = axes[1, 1]
    ax.axis('off')
    
    dist_text = "📏 DEPTH PROFILE\n" + "━"*25 + "\n"
    if 'distance_performance' in card and 'Avg Conv' in card['distance_performance']:
        for dist, conv in card['distance_performance']['Avg Conv'].items():
            dist_text += f"• {dist}: {conv:.3f} yd/s\n"
    
    ax.text(0.1, 0.5, dist_text, fontsize=10, family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

print("\n" + "="*80)
print("GENERATING VISUAL RECEIVER CARDS")
print("="*80)

cards_dir = OUTPUT_DIR / 'receiver_cards'
cards_dir.mkdir(exist_ok=True)

for player_id, card in receiver_cards.items():
    if 'player_name' in card:
        clean_name = card['player_name'].replace(' ', '_').replace('.', '')
        team = card.get('team', 'UNK')
        card_file = cards_dir / f'{clean_name}_{team}_{int(player_id)}.png'
    else:
        card_file = cards_dir / f'receiver_{int(player_id)}_card.png'
    
    create_visual_receiver_card(player_id, card, card_file)

print(f"✓ Generated {len(receiver_cards)} visual cards")
print(f"✅ Cards saved to: {cards_dir}")

# %% [markdown]
# ## 9. Summary Report

# %%
print("\n" + "="*80)
print("OFFENSIVE PLAYER MATCHUP ANALYSIS - SUMMARY")
print("="*80)

print(f"""
ANALYSIS COMPLETE:

OFFENSIVE PLAYERS ANALYZED:
- Total receivers/targets: {offense['nfl_id'].nunique():,}
- Receivers with coverage data: {receiver_coverage_perf['nfl_id'].nunique():,}
- Receivers with route data: {receiver_route_perf['nfl_id'].nunique():,}
- Receivers with distance data: {receiver_distance_perf['nfl_id'].nunique():,}
- Priority scouting cards generated: {len(receiver_cards)}

TOP 3 RECEIVERS WHO DOMINATE SPECIFIC COVERAGES:
""")

for idx, (i, row) in enumerate(cov_matchup_df.head(3).iterrows(), 1):
    if 'player_name' in cov_matchup_df.columns:
        player_str = f"{row['player_name']} ({row.get('team', 'UNK')}) - {row['position']}"
    else:
        player_str = f"Player #{int(row['nfl_id'])}"
    
    print(f"  {idx}. {player_str}")
    print(f"     Dominates: {row['best_coverage']} ({row['best_coverage_conv']:.3f} yd/s)")
    print(f"     Struggles vs: {row['worst_coverage']} ({row['worst_coverage_conv']:.3f} yd/s)")
    print(f"     Coverage gap: {row['coverage_gap']:.3f} yd/s\n")

print(f"""
TOP 3 ROUTE SPECIALISTS:
""")

for idx, (i, row) in enumerate(route_strength_df.head(3).iterrows(), 1):
    if 'player_name' in route_strength_df.columns:
        player_str = f"{row['player_name']} ({row.get('team', 'UNK')}) - {row['position']}"
    else:
        player_str = f"Player #{int(row['nfl_id'])}"
    
    print(f"  {idx}. {player_str}")
    print(f"     Best route: {row['best_route']} ({row['best_route_conv']:.3f} yd/s)")
    print(f"     Worst route: {row['worst_route']} ({row['worst_route_conv']:.3f} yd/s)")
    print(f"     Route gap: {row['route_gap']:.3f} yd/s\n")

print(f"""
FILES GENERATED:
✓ receiver_coverage_matchups.csv - Coverage-specific performance
✓ receiver_route_strengths.csv - Route-specific performance  
✓ receiver_vs_defender_position.csv - Position matchup data
✓ receiver_depth_strengths.csv - Distance/depth performance
✓ receiver_coverage_heatmap.png - Visual matrix
✓ receiver_coverage_matrix.csv - Performance matrix
✓ receiver_cards/ - Individual receiver scouting cards

USAGE:
1. Identify opponent's defensive weaknesses (coverage types, positions)
2. Match your receivers' strengths to opponent's weaknesses
3. Design route concepts that exploit coverage matchups
4. Use depth analysis to optimize route combinations
""")

print("="*80)
print("✅ OFFENSIVE MATCHUP ANALYSIS COMPLETE!")
print("="*80)