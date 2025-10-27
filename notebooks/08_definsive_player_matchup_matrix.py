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
# # NFL Big Data Bowl 2026 - Player Matchup Matrix

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
DATA_DIR = Path('../data')
PROCESSED_DIR = DATA_DIR / 'processed'
COMPETITION_DIR = DATA_DIR / '114239_nfl_competition_files_published_analytics_final'
TRAIN_DIR = COMPETITION_DIR / 'train'
OUTPUT_DIR = PROCESSED_DIR / 'definsive_player_matchups'
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

defenders = merged_df[merged_df['player_role'] == 'Defensive Coverage'].copy()
print(f"✓ Defender instances: {len(defenders):,}")

# %%
# LOAD PLAYER NAMES FROM ALL INPUT FILES (MODIFIED)
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
        # Combine all player data and keep first occurrence of each nfl_id
        player_lookup = pd.concat(all_players, ignore_index=True).drop_duplicates('nfl_id', keep='first')
        player_lookup.columns = ['nfl_id', 'player_name', 'position']
        
        print(f"\n✅ TOTAL UNIQUE PLAYERS LOADED: {len(player_lookup)}")
        
        # Add team info
        if 'defensive_team' in defenders.columns:
            team_lookup = defenders.groupby('nfl_id')['defensive_team'].agg(
                lambda x: x.mode()[0] if len(x.mode()) > 0 else 'UNK'
            ).reset_index()
            team_lookup.columns = ['nfl_id', 'team']
            player_lookup = player_lookup.merge(team_lookup, on='nfl_id', how='left')
            player_lookup['team'] = player_lookup['team'].fillna('UNK')
            print(f"✓ Added team info")
        
        # CRITICAL: Filter defenders to only include players we have names for
        print(f"\nBefore filtering: {defenders['nfl_id'].nunique()} unique defenders")
        defenders = defenders[defenders['nfl_id'].isin(player_lookup['nfl_id'])].copy()
        print(f"After filtering: {defenders['nfl_id'].nunique()} unique defenders")
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
# ## 2. Player-Route Performance Matrix

# %%
print("="*80)
print("PLAYER × ROUTE PERFORMANCE ANALYSIS")
print("="*80)

MIN_PLAYS_ROUTE = 3

player_route_perf = defenders.groupby(['nfl_id', 'route_of_targeted_receiver']).agg({
    'convergence_speed': ['mean', 'std', 'count'],
    'min_distance': 'mean',
    'player_position': lambda x: x.mode()[0] if len(x) > 0 else None
}).round(3)

player_route_perf.columns = ['Avg Conv', 'Std Dev', 'N Plays', 'Avg Min Dist', 'Position']
player_route_perf = player_route_perf[player_route_perf['N Plays'] >= MIN_PLAYS_ROUTE].reset_index()

print(f"\nPlayers with route-specific data: {player_route_perf['nfl_id'].nunique()}")
print(f"Total player-route combinations: {len(player_route_perf)}")

# Identify biggest route vulnerabilities
print("\n" + "="*80)
print("TOP PLAYER-ROUTE VULNERABILITIES TO EXPLOIT")
print("="*80)

player_vulnerabilities = []

for player_id in player_route_perf['nfl_id'].unique():
    player_data = player_route_perf[player_route_perf['nfl_id'] == player_id]
    
    if len(player_data) < 2:
        continue
    
    worst_route_idx = player_data['Avg Conv'].idxmin()
    worst_route = player_data.loc[worst_route_idx]
    
    best_route_idx = player_data['Avg Conv'].idxmax()
    best_route = player_data.loc[best_route_idx]
    
    vulnerability_score = best_route['Avg Conv'] - worst_route['Avg Conv']
    
    player_vulnerabilities.append({
        'nfl_id': player_id,
        'worst_route': worst_route['route_of_targeted_receiver'],
        'worst_route_conv': worst_route['Avg Conv'],
        'worst_route_plays': worst_route['N Plays'],
        'best_route': best_route['route_of_targeted_receiver'],
        'best_route_conv': best_route['Avg Conv'],
        'vulnerability_score': vulnerability_score
    })

vuln_df = pd.DataFrame(player_vulnerabilities)
vuln_df = vuln_df.sort_values('vulnerability_score', ascending=False)

# Add player names and teams (MODIFIED WITH NAN HANDLING)
if player_lookup is not None:
    vuln_df = vuln_df.merge(
        player_lookup[['nfl_id', 'player_name', 'team', 'position']], 
        on='nfl_id', 
        how='left'
    )
    
    # Handle any remaining NaN values (shouldn't happen after filtering, but just in case)
    vuln_df['player_name'] = vuln_df['player_name'].fillna('Unknown Player')
    vuln_df['team'] = vuln_df['team'].fillna('UNK')
    vuln_df['position'] = vuln_df['position'].fillna('UNK')
    
    # Reorder columns
    cols = ['nfl_id', 'player_name', 'team', 'position', 'worst_route', 'worst_route_conv', 
            'worst_route_plays', 'best_route', 'best_route_conv', 'vulnerability_score']
    vuln_df = vuln_df[cols]

print("\nTop 20 Players with Biggest Route Vulnerabilities:")
print("-" * 80)
if player_lookup is not None and 'player_name' in vuln_df.columns:
    display_cols = ['player_name', 'team', 'position', 'worst_route', 'worst_route_conv', 
                    'best_route', 'best_route_conv', 'vulnerability_score']
    print(vuln_df[display_cols].head(20).to_string(index=False))
else:
    print(vuln_df.head(20).to_string(index=False))

vuln_df.to_csv(OUTPUT_DIR / 'definsive_player_route_vulnerabilities.csv', index=False)

# %% [markdown]
# ## 3. Player Performance by Coverage

# %%
print("\n" + "="*80)
print("PLAYER × COVERAGE PERFORMANCE ANALYSIS")
print("="*80)

MIN_PLAYS_COV = 5

player_coverage_perf = defenders.groupby(['nfl_id', 'team_coverage_type']).agg({
    'convergence_speed': ['mean', 'count'],
    'min_distance': 'mean',
    'player_position': lambda x: x.mode()[0] if len(x) > 0 else None
}).round(3)

player_coverage_perf.columns = ['Avg Conv', 'N Plays', 'Avg Min Dist', 'Position']
player_coverage_perf = player_coverage_perf[player_coverage_perf['N Plays'] >= MIN_PLAYS_COV].reset_index()

print(f"\nPlayers with coverage-specific data: {player_coverage_perf['nfl_id'].nunique()}")

coverage_vulnerabilities = []

for player_id in player_coverage_perf['nfl_id'].unique():
    player_data = player_coverage_perf[player_coverage_perf['nfl_id'] == player_id]
    
    if len(player_data) < 2:
        continue
    
    worst_cov_idx = player_data['Avg Conv'].idxmin()
    worst_cov = player_data.loc[worst_cov_idx]
    
    best_cov_idx = player_data['Avg Conv'].idxmax()
    best_cov = player_data.loc[best_cov_idx]
    
    cov_diff = best_cov['Avg Conv'] - worst_cov['Avg Conv']
    
    coverage_vulnerabilities.append({
        'nfl_id': player_id,
        'worst_coverage': worst_cov['team_coverage_type'],
        'worst_coverage_conv': worst_cov['Avg Conv'],
        'best_coverage': best_cov['team_coverage_type'],
        'best_coverage_conv': best_cov['Avg Conv'],
        'coverage_diff': cov_diff
    })

cov_vuln_df = pd.DataFrame(coverage_vulnerabilities)
cov_vuln_df = cov_vuln_df.sort_values('coverage_diff', ascending=False)

# Add player names and teams (MODIFIED WITH NAN HANDLING)
if player_lookup is not None:
    cov_vuln_df = cov_vuln_df.merge(
        player_lookup[['nfl_id', 'player_name', 'team', 'position']], 
        on='nfl_id', 
        how='left'
    )
    
    # Handle any remaining NaN values
    cov_vuln_df['player_name'] = cov_vuln_df['player_name'].fillna('Unknown Player')
    cov_vuln_df['team'] = cov_vuln_df['team'].fillna('UNK')
    cov_vuln_df['position'] = cov_vuln_df['position'].fillna('UNK')
    
    cols = ['nfl_id', 'player_name', 'team', 'position', 'worst_coverage', 'worst_coverage_conv', 
            'best_coverage', 'best_coverage_conv', 'coverage_diff']
    cov_vuln_df = cov_vuln_df[cols]

print("\nTop 20 Players with Coverage-Specific Weaknesses:")
print("-" * 80)
if player_lookup is not None and 'player_name' in cov_vuln_df.columns:
    display_cols = ['player_name', 'team', 'position', 'worst_coverage', 'worst_coverage_conv', 
                    'best_coverage', 'best_coverage_conv', 'coverage_diff']
    print(cov_vuln_df[display_cols].head(20).to_string(index=False))
else:
    print(cov_vuln_df.head(20).to_string(index=False))

cov_vuln_df.to_csv(OUTPUT_DIR / 'definsive_player_coverage_vulnerabilities.csv', index=False)

# %% [markdown]
# ## 4. Distance-Based Performance

# %%
print("\n" + "="*80)
print("PLAYER × STARTING DISTANCE ANALYSIS")
print("="*80)

defenders['distance_category'] = pd.cut(
    defenders['initial_distance'],
    bins=[0, 10, 15, 20, 100],
    labels=['Close (<10y)', 'Medium (10-15y)', 'Far (15-20y)', 'Very Far (>20y)']
)

MIN_PLAYS_DIST = 5

player_distance_perf = defenders.groupby(['nfl_id', 'distance_category']).agg({
    'convergence_speed': ['mean', 'count'],
    'min_distance': 'mean',
    'player_position': lambda x: x.mode()[0] if len(x) > 0 else None
}).round(3)

player_distance_perf.columns = ['Avg Conv', 'N Plays', 'Avg Min Dist', 'Position']
player_distance_perf = player_distance_perf[player_distance_perf['N Plays'] >= MIN_PLAYS_DIST].reset_index()

print(f"\nPlayers with distance-specific data: {player_distance_perf['nfl_id'].nunique()}")

distance_vulnerabilities = []

for player_id in player_distance_perf['nfl_id'].nunique():
    player_data = player_distance_perf[player_distance_perf['nfl_id'] == player_id]
    
    if len(player_data) < 2:
        continue
    
    worst_dist_idx = player_data['Avg Conv'].idxmin()
    worst_dist = player_data.loc[worst_dist_idx]
    
    best_dist_idx = player_data['Avg Conv'].idxmax()
    best_dist = player_data.loc[best_dist_idx]
    
    dist_diff = best_dist['Avg Conv'] - worst_dist['Avg Conv']
    
    distance_vulnerabilities.append({
        'nfl_id': player_id,
        'worst_distance': worst_dist['distance_category'],
        'worst_distance_conv': worst_dist['Avg Conv'],
        'best_distance': best_dist['distance_category'],
        'best_distance_conv': best_dist['Avg Conv'],
        'distance_diff': dist_diff
    })

dist_vuln_df = pd.DataFrame(distance_vulnerabilities)
dist_vuln_df = dist_vuln_df.sort_values('distance_diff', ascending=False)

# Add player names and teams (MODIFIED WITH NAN HANDLING)
if player_lookup is not None:
    dist_vuln_df = dist_vuln_df.merge(
        player_lookup[['nfl_id', 'player_name', 'team', 'position']], 
        on='nfl_id', 
        how='left'
    )
    
    # Handle any remaining NaN values
    dist_vuln_df['player_name'] = dist_vuln_df['player_name'].fillna('Unknown Player')
    dist_vuln_df['team'] = dist_vuln_df['team'].fillna('UNK')
    dist_vuln_df['position'] = dist_vuln_df['position'].fillna('UNK')
    
    cols = ['nfl_id', 'player_name', 'team', 'position', 'worst_distance', 'worst_distance_conv', 
            'best_distance', 'best_distance_conv', 'distance_diff']
    dist_vuln_df = dist_vuln_df[cols]

print("\nTop 20 Players with Distance-Specific Weaknesses:")
print("-" * 80)
if player_lookup is not None and 'player_name' in dist_vuln_df.columns:
    display_cols = ['player_name', 'team', 'position', 'worst_distance', 'worst_distance_conv', 
                    'best_distance', 'best_distance_conv', 'distance_diff']
    print(dist_vuln_df[display_cols].head(20).to_string(index=False))
else:
    print(dist_vuln_df.head(20).to_string(index=False))

dist_vuln_df.to_csv(OUTPUT_DIR / 'definsive_player_distance_vulnerabilities.csv', index=False)

# %% [markdown]
# ## 5. Comprehensive Player Scouting Card

# %%
def generate_player_scouting_card(player_id, data, player_lookup=None):
    """Generate comprehensive player scouting report"""
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
    
    route_perf = player_data.groupby('route_of_targeted_receiver').agg({
        'convergence_speed': 'mean',
        'game_id': 'count'
    }).round(3)
    route_perf.columns = ['Avg Conv', 'N']
    route_perf = route_perf[route_perf['N'] >= 3].sort_values('Avg Conv')
    
    card['worst_routes'] = route_perf.head(3).to_dict()
    card['best_routes'] = route_perf.tail(3).to_dict()
    
    if 'team_coverage_type' in player_data.columns:
        cov_perf = player_data.groupby('team_coverage_type').agg({
            'convergence_speed': 'mean',
            'game_id': 'count'
        }).round(3)
        cov_perf.columns = ['Avg Conv', 'N']
        cov_perf = cov_perf[cov_perf['N'] >= 3].sort_values('Avg Conv')
        
        card['coverage_performance'] = cov_perf.to_dict()
    
    player_data_dist = player_data.copy()
    player_data_dist['dist_cat'] = pd.cut(
        player_data_dist['initial_distance'],
        bins=[0, 10, 15, 20, 100],
        labels=['<10y', '10-15y', '15-20y', '>20y']
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
print("GENERATING INDIVIDUAL PLAYER SCOUTING CARDS")
print("="*80)

top_route_vuln = vuln_df.head(10)['nfl_id'].tolist()
top_cov_vuln = cov_vuln_df.head(10)['nfl_id'].tolist()
top_dist_vuln = dist_vuln_df.head(10)['nfl_id'].tolist()

priority_players = list(set(top_route_vuln + top_cov_vuln + top_dist_vuln))

player_cards = {}

for player_id in priority_players:
    card = generate_player_scouting_card(player_id, defenders, player_lookup)
    if card:
        player_cards[player_id] = card
        name_str = f"{card.get('player_name', f'Player #{player_id}')}"
        team_str = f" ({card.get('team', 'UNK')})" if 'team' in card else ""
        pos_str = f" - {card['position']}"
        print(f"✓ Generated card for {name_str}{team_str}{pos_str}")

print(f"\n✅ Generated {len(player_cards)} player scouting cards")

# %% [markdown]
# ## 6. Visualize Player Matchup Matrices

# %%
print("\n" + "="*80)
print("CREATING PLAYER-ROUTE HEATMAP")
print("="*80)

top_players = vuln_df.head(20)['nfl_id'].tolist()

player_route_matrix = player_route_perf[player_route_perf['nfl_id'].isin(top_players)]

# Add player names with NaN handling (MODIFIED)
if player_lookup is not None:
    player_route_matrix = player_route_matrix.merge(
        player_lookup[['nfl_id', 'player_name', 'team']], 
        on='nfl_id', 
        how='left'
    )
    
    # Handle NaN values in player_name and team
    player_route_matrix['player_name'] = player_route_matrix['player_name'].fillna('Unknown')
    player_route_matrix['team'] = player_route_matrix['team'].fillna('UNK')
    
    player_route_matrix['player_label'] = player_route_matrix.apply(
        lambda x: f"{x['player_name']} ({x['team']})", 
        axis=1
    )
    index_col = 'player_label'
else:
    player_route_matrix['player_label'] = player_route_matrix['nfl_id'].apply(lambda x: f"Player #{int(x)}")
    index_col = 'player_label'

matrix_pivot = player_route_matrix.pivot_table(
    values='Avg Conv',
    index=index_col,
    columns='route_of_targeted_receiver',
    aggfunc='mean'
)

if len(matrix_pivot) > 0:
    matrix_pivot.to_csv(OUTPUT_DIR / 'defensive_player_route_vulnerability_matrix.csv')
    
    print(f"\n✅ Player-Route Matrix Created ({len(matrix_pivot)} players × {len(matrix_pivot.columns)} routes)")
    print("\nTop 10 players in matrix:")
    print(matrix_pivot.head(10))
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    sns.heatmap(matrix_pivot, annot=True, fmt='.2f', cmap='RdYlGn', 
               center=0, ax=ax, cbar_kws={'label': 'Avg Convergence (yd/s)'})
    
    ax.set_title('Player-Route Vulnerability Matrix (Top 20 Vulnerable Players)\n' + 
                'Lower values (red) = Attack these routes against these players',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Route Type', fontsize=12)
    ax.set_ylabel('Player Name (Team)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'player_route_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Heatmap saved!")
else:
    print("⚠️ No data for heatmap")

# %% [markdown]
# ## 7. Player Quick Reference Cards (Visual)

# %%
def create_visual_player_card(player_id, card, filename):
    """Create visual quick reference card for a player"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    player_name = card.get('player_name', f'Player #{player_id}')
    team = card.get('team', 'UNK')
    position = card['position']
    
    title = f'{player_name} ({team}) - {position}\nSCOUTING CARD'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    ax = axes[0, 0]
    ax.axis('off')
    
    stats_text = f"""
OVERALL STATS
━━━━━━━━━━━━━━━━━━
Total Plays: {card['total_plays']}
Avg Convergence: {card['avg_convergence']:.3f} yd/s
Avg Min Distance: {card['avg_min_distance']:.2f} yards
Avg Speed: {card['avg_speed']:.2f} yd/s
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax = axes[0, 1]
    ax.axis('off')
    
    routes_text = "🎯 ROUTES TO ATTACK\n" + "━"*25 + "\n"
    if 'worst_routes' in card and 'Avg Conv' in card['worst_routes']:
        for route, conv in list(card['worst_routes']['Avg Conv'].items())[:3]:
            n = card['worst_routes']['N'].get(route, 0)
            routes_text += f"• {route}: {conv:.3f} yd/s ({int(n)})\n"
    
    ax.text(0.1, 0.5, routes_text, fontsize=10, family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    ax = axes[1, 0]
    ax.axis('off')
    
    cov_text = "📊 COVERAGE NOTES\n" + "━"*25 + "\n"
    if 'coverage_performance' in card and 'Avg Conv' in card['coverage_performance']:
        for cov, conv in list(card['coverage_performance']['Avg Conv'].items())[:3]:
            cov_text += f"• {cov}: {conv:.3f} yd/s\n"
    
    ax.text(0.1, 0.5, cov_text, fontsize=10, family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    ax = axes[1, 1]
    ax.axis('off')
    
    dist_text = "📏 DISTANCE PROFILE\n" + "━"*25 + "\n"
    if 'distance_performance' in card and 'Avg Conv' in card['distance_performance']:
        for dist, conv in card['distance_performance']['Avg Conv'].items():
            dist_text += f"• {dist}: {conv:.3f} yd/s\n"
    
    ax.text(0.1, 0.5, dist_text, fontsize=10, family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

print("\n" + "="*80)
print("GENERATING VISUAL PLAYER CARDS")
print("="*80)

cards_dir = OUTPUT_DIR / 'definsive_cards'
cards_dir.mkdir(exist_ok=True)

for player_id, card in player_cards.items():
    if 'player_name' in card:
        clean_name = card['player_name'].replace(' ', '_').replace('.', '')
        team = card.get('team', 'UNK')
        card_file = cards_dir / f'{clean_name}_{team}_{int(player_id)}.png'
    else:
        card_file = cards_dir / f'player_{int(player_id)}_card.png'
    
    create_visual_player_card(player_id, card, card_file)

print(f"✓ Generated {len(player_cards)} visual cards")
print(f"✅ Cards saved to: {cards_dir}")

# %% [markdown]
# ## 8. Summary Report

# %%
print("\n" + "="*80)
print("PLAYER MATCHUP ANALYSIS - SUMMARY")
print("="*80)

print(f"""
ANALYSIS COMPLETE:

PLAYERS ANALYZED:
- Total defenders: {defenders['nfl_id'].nunique():,}
- Players with route data: {player_route_perf['nfl_id'].nunique():,}
- Players with coverage data: {player_coverage_perf['nfl_id'].nunique():,}
- Players with distance data: {player_distance_perf['nfl_id'].nunique():,}
- Priority scouting cards generated: {len(player_cards)}

TOP 3 BIGGEST ROUTE VULNERABILITIES:
""")

for idx, (i, row) in enumerate(vuln_df.head(3).iterrows(), 1):
    if 'player_name' in vuln_df.columns:
        player_str = f"{row['player_name']} ({row.get('team', 'UNK')}) - {row['position']}"
    else:
        player_str = f"Player #{int(row['nfl_id'])}"
    
    print(f"  {idx}. {player_str}")
    print(f"     Worst on: {row['worst_route']} ({row['worst_route_conv']:.3f} yd/s)")
    print(f"     Best on: {row['best_route']} ({row['best_route_conv']:.3f} yd/s)")
    print(f"     Vulnerability gap: {row['vulnerability_score']:.3f} yd/s\n")

print(f"""
TOP 3 COVERAGE-SPECIFIC WEAKNESSES:
""")

for idx, (i, row) in enumerate(cov_vuln_df.head(3).iterrows(), 1):
    if 'player_name' in cov_vuln_df.columns:
        player_str = f"{row['player_name']} ({row.get('team', 'UNK')}) - {row['position']}"
    else:
        player_str = f"Player #{int(row['nfl_id'])}"
    
    print(f"  {idx}. {player_str}")
    print(f"     Worst in: {row['worst_coverage']} ({row['worst_coverage_conv']:.3f} yd/s)")
    print(f"     Best in: {row['best_coverage']} ({row['best_coverage_conv']:.3f} yd/s)")
    print(f"     Coverage gap: {row['coverage_diff']:.3f} yd/s\n")

print(f"""
FILES GENERATED:
✓ player_route_vulnerabilities.csv - Route-specific weaknesses WITH NAMES
✓ player_coverage_vulnerabilities.csv - Coverage-specific weaknesses WITH NAMES
✓ player_distance_vulnerabilities.csv - Distance-specific weaknesses WITH NAMES
✓ player_route_heatmap.png - Visual matrix WITH NAMES
✓ player_route_vulnerability_matrix.csv - Matrix WITH NAMES
✓ player_cards/ - Individual player scouting cards WITH NAMES

USAGE:
1. Identify opponent's defenders from team roster
2. Check their vulnerability files for specific weaknesses
3. Use visual cards for quick reference during game planning
4. Design plays to exploit identified route/coverage mismatches
""")

print("="*80)
print("✅ PLAYER MATCHUP ANALYSIS COMPLETE!")
print("="*80)