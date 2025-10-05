# NFL Big Data Bowl 2026 - Analytics Track

## Project Overview

This project analyzes NFL player movement during pass plays for the 2026 Big Data Bowl Analytics Competition. The focus is on understanding how players move and react while the ball is in the air, from the moment the quarterback releases the ball until it's caught or ruled incomplete.

## Competition Details

- **Competition**: NFL Big Data Bowl 2026 - Analytics Track
- **Timeline**: September 25, 2025 - December 17, 2025
- **Submission Deadline**: December 17, 2025 (11:59 PM UTC)
- **Competition Link**: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics

## Project Structure

```
nfl-bdb-2026/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore file
├── notebooks/
│   ├── 01_data_download.ipynb  # Download data from Kaggle
│   ├── 02_exploration.ipynb    # Initial data exploration
│   ├── 03_analysis.ipynb       # Core analysis and metrics
│   └── 04_visualization.ipynb  # Create final visualizations
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading utilities
│   ├── metrics.py              # Custom metric calculations
│   └── visualization.py        # Plotting functions
├── data/                        # Data directory (gitignored)
│   ├── tracking.csv
│   ├── plays.csv
│   ├── players.csv
│   └── games.csv
├── figures/                     # Output visualizations
└── writeup/
    └── submission.md            # Draft writeup
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd nfl-bdb-2026
```

### 2. Create Python Environment

**Using Conda (Recommended):**
```bash
conda create -n nfl-bdb python=3.10
conda activate nfl-bdb
```

**Using venv:**
```bash
python -m venv nfl-bdb
source nfl-bdb/bin/activate  # Mac/Linux
# OR
nfl-bdb\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Kaggle API

To download data directly from Kaggle, you need to set up your Kaggle API credentials:

1. Go to your Kaggle account settings: https://www.kaggle.com/settings
2. Scroll to "API" section and click "Create New Token"
3. This downloads `kaggle.json` file
4. Place the file in the correct location:

**Mac/Linux:**
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Windows:**
```bash
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\kaggle.json
```

### 5. Accept Competition Rules

Before downloading data, you must accept the competition rules:
1. Go to: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics
2. Click "Join Competition" or "I Understand and Accept"

### 6. Download Data

Run the first notebook to download data:
```bash
jupyter notebook notebooks/01_data_download.ipynb
```

Or use the command line:
```bash
kaggle competitions download -c nfl-big-data-bowl-2026-analytics
unzip nfl-big-data-bowl-2026-analytics.zip -d data/
```

## Data Description

### Tracking Data
- Player positions at 10 frames per second
- Includes x, y coordinates, speed, acceleration, direction
- Ball tracking included

### Plays Data
- Play-level information
- Down, distance, formation
- Pass result and target receiver
- **Key**: `passForwardFrameId` and `passOutcomeFrameId`

### Players Data
- Player demographics
- Position, height, weight

### Games Data
- Game metadata
- Teams, weather, stadium info

## Analysis Focus

This project analyzes: **[DESCRIBE YOUR CHOSEN METRIC/ANALYSIS HERE]**

Example: "Defensive Convergence Speed - measuring how quickly defenders react and close distance to the target receiver after the ball is thrown."

## Key Metrics

1. **[Metric 1 Name]**: [Brief description]
2. **[Metric 2 Name]**: [Brief description]
3. **[Metric 3 Name]**: [Brief description]

## Usage

### Explore Data
```bash
jupyter notebook notebooks/02_exploration.ipynb
```

### Run Analysis
```bash
jupyter notebook notebooks/03_analysis.ipynb
```

### Generate Visualizations
```bash
jupyter notebook notebooks/04_visualization.ipynb
```

## Key Findings

*[Update this section as you progress]*

1. Finding 1...
2. Finding 2...
3. Finding 3...

## Visualizations

Key visualizations in `figures/` directory:
- `figure1_[description].png`
- `figure2_[description].png`
- etc.

## Submission Requirements

- [x] Kaggle Writeup (under 2000 words)
- [x] Media Gallery with cover image
- [ ] Public notebook attached
- [ ] Less than 10 figures/tables
- [ ] Video (if Broadcast Track, 3 min max)

## Timeline

- **Week 1-2** (Oct 2-15): Setup & Exploration ✅
- **Week 3-4** (Oct 16-29): Metric Development
- **Week 5-6** (Oct 30 - Nov 12): Analysis & Insights
- **Week 7-8** (Nov 13-26): Visualization Development
- **Week 9** (Nov 27 - Dec 3): Writeup & Assembly
- **Week 10** (Dec 4-17): Video & Final Polish

## Resources

- **Competition Page**: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics
- **NFL Next Gen Stats**: https://nextgenstats.nfl.com/
- **Past Winners**: https://operations.nfl.com/gameday/analytics/big-data-bowl/
- **Kaggle API Docs**: https://github.com/Kaggle/kaggle-api

## Contributing

This is a solo competition project, but feedback is welcome!

## License

This project is for the NFL Big Data Bowl 2026 competition.

## Acknowledgments

- NFL & AWS for organizing the Big Data Bowl
- Kaggle for hosting the competition
- Next Gen Stats team for the incredible data

## Contact

Jean Paul
jp86miter@gmail.com
[Kaggle Profile](https://www.kaggle.com/jeanpaulm)

---
