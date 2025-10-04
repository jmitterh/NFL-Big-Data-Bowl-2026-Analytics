@echo off
REM NFL Big Data Bowl 2026 - Windows Setup Script

echo.
echo ========================================
echo NFL Big Data Bowl 2026 - Setup Script
echo ========================================
echo.

REM Create directory structure
echo Creating project directories...
if not exist "data" mkdir data
if not exist "notebooks" mkdir notebooks
if not exist "src" mkdir src
if not exist "figures" mkdir figures
if not exist "writeup" mkdir writeup

REM Create __init__.py for src package
type nul > src\__init__.py

REM Create data_loader.py
echo Creating src\data_loader.py...
(
echo """
echo Data loading utilities for NFL Big Data Bowl 2026
echo """
echo import pandas as pd
echo from pathlib import Path
echo.
echo DATA_DIR = Path^(__file__^).parent.parent / 'data'
echo.
echo def load_tracking^(^):
echo     """Load tracking data"""
echo     return pd.read_csv^(DATA_DIR / 'tracking.csv'^)
echo.
echo def load_plays^(^):
echo     """Load plays data"""
echo     return pd.read_csv^(DATA_DIR / 'plays.csv'^)
echo.
echo def load_players^(^):
echo     """Load players data"""
echo     return pd.read_csv^(DATA_DIR / 'players.csv'^)
echo.
echo def load_games^(^):
echo     """Load games data"""
echo     return pd.read_csv^(DATA_DIR / 'games.csv'^)
echo.
echo def load_all^(^):
echo     """Load all datasets"""
echo     return {
echo         'tracking': load_tracking^(^),
echo         'plays': load_plays^(^),
echo         'players': load_players^(^),
echo         'games': load_games^(^)
echo     }
) > src\data_loader.py

REM Create metrics.py
echo Creating src\metrics.py...
(
echo """
echo Custom metrics and calculations
echo """
echo import pandas as pd
echo import numpy as np
echo.
echo def calculate_distance^(x1, y1, x2, y2^):
echo     """Calculate Euclidean distance between two points"""
echo     return np.sqrt^(^(x2 - x1^)**2 + ^(y2 - y1^)**2^)
echo.
echo def get_ball_in_air_frames^(tracking_df, plays_df^):
echo     """Filter tracking data to only frames where ball is in the air"""
echo     merged = tracking_df.merge^(
echo         plays_df[['gameId', 'playId', 'passForwardFrameId', 'passOutcomeFrameId']], 
echo         on=['gameId', 'playId'],
echo         how='inner'
echo     ^)
echo     ball_in_air = merged[
echo         ^(merged['frameId'] ^>= merged['passForwardFrameId']^) ^& 
echo         ^(merged['frameId'] ^<= merged['passOutcomeFrameId']^)
echo     ]
echo     return ball_in_air
) > src\metrics.py

REM Create visualization.py
echo Creating src\visualization.py...
(
echo """
echo Visualization utilities
echo """
echo import matplotlib.pyplot as plt
echo import seaborn as sns
echo.
echo def setup_plot_style^(^):
echo     """Set consistent plot style"""
echo     sns.set_style^("whitegrid"^)
echo     plt.rcParams['figure.figsize'] = ^(12, 8^)
echo     plt.rcParams['font.size'] = 10
echo.
echo def plot_field^(^):
echo     """Draw NFL field boundaries"""
echo     fig, ax = plt.subplots^(figsize=^(12, 6^)^)
echo     ax.set_xlim^(0, 120^)
echo     ax.set_ylim^(0, 53.3^)
echo     ax.set_xlabel^('X Position ^(yards^)'^)
echo     ax.set_ylabel^('Y Position ^(yards^)'^)
echo     ax.grid^(alpha=0.3^)
echo     return fig, ax
) > src\visualization.py

echo.
echo Directory structure created successfully!
echo.

REM Check if requirements.txt exists and install
if exist "requirements.txt" (
    echo Installing Python packages...
    pip install -r requirements.txt
    echo Packages installed!
) else (
    echo WARNING: requirements.txt not found. Please create it first.
)

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo Next steps:
echo 1. Set up your Kaggle API credentials ^(kaggle.json^)
echo 2. Run: jupyter notebook notebooks\01_data_download.ipynb
echo 3. Start exploring the data!
echo.
echo Good luck! 
echo.
pause