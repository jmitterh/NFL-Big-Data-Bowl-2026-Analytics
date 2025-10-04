#!/bin/bash

# NFL Big Data Bowl 2026 - Setup Script
# This script sets up your project structure and environment

echo "🏈 NFL Big Data Bowl 2026 - Setup Script"
echo "========================================="
echo ""

# Create directory structure
echo "📁 Creating project directories..."
mkdir -p data
mkdir -p notebooks
mkdir -p src
mkdir -p figures
mkdir -p writeup

# Create __init__.py for src package
touch src/__init__.py

# Create placeholder files in src/
cat > src/data_loader.py << 'EOF'
"""
Data loading utilities for NFL Big Data Bowl 2026
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'

def load_tracking():
    """Load tracking data"""
    return pd.read_csv(DATA_DIR / 'tracking.csv')

def load_plays():
    """Load plays data"""
    return pd.read_csv(DATA_DIR / 'plays.csv')

def load_players():
    """Load players data"""
    return pd.read_csv(DATA_DIR / 'players.csv')

def load_games():
    """Load games data"""
    return pd.read_csv(DATA_DIR / 'games.csv')

def load_all():
    """Load all datasets"""
    return {
        'tracking': load_tracking(),
        'plays': load_plays(),
        'players': load_players(),
        'games': load_games()
    }
EOF

cat > src/metrics.py << 'EOF'
"""
Custom metrics and calculations
"""
import pandas as pd
import numpy as np

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_ball_in_air_frames(tracking_df, plays_df):
    """
    Filter tracking data to only frames where ball is in the air
    """
    merged = tracking_df.merge(
        plays_df[['gameId', 'playId', 'passForwardFrameId', 'passOutcomeFrameId']], 
        on=['gameId', 'playId'],
        how='inner'
    )
    
    ball_in_air = merged[
        (merged['frameId'] >= merged['passForwardFrameId']) & 
        (merged['frameId'] <= merged['passOutcomeFrameId'])
    ]
    
    return ball_in_air
EOF

cat > src/visualization.py << 'EOF'
"""
Visualization utilities
"""
import matplotlib.pyplot as plt
import seaborn as sns

def setup_plot_style():
    """Set consistent plot style"""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10

def plot_field():
    """Draw NFL field boundaries"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_xlabel('X Position (yards)')
    ax.set_ylabel('Y Position (yards)')
    ax.grid(alpha=0.3)
    return fig, ax
EOF

echo "✅ Directory structure created!"
echo ""

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing Python packages..."
    pip install -r requirements.txt
    echo "✅ Packages installed!"
else
    echo "⚠️  requirements.txt not found. Please create it first."
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Set up your Kaggle API credentials (kaggle.json)"
echo "2. Run: jupyter notebook notebooks/01_data_download.ipynb"
echo "3. Start exploring the data!"
echo ""
echo "Good luck! 🏈"