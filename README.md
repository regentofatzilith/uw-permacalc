# UW PermaCalc

**Ultimate Weapon Permanence Calculator** for The Tower game.

A comprehensive calculator for analyzing Ultimate Weapon uptime, cooldowns, and optimal configurations across different game modes and equipment setups.

## Live Demo

🚀 **[Try it here](https://your-app-name.onrender.com)** (replace with your actual URL after deployment)

## Features

### Ultimate Weapons Supported
- ⚫ **Black Hole** - Mass gravity control
- 🟡 **Golden Tower** - Income booster
- 🔴 **Death Wave** - Area damage dealer
- 🔵 **Chrono Field** - Time manipulation
- 🟡 **Golden Bot** - Resource multiplier

### Configuration Options
- 🎮 **Game Modes**: Tournament / Farming
- 🃏 **Wave Accelerator Card**: None to 7 Star (wave speed boost)
- 👹 **Boss Wave Frequency**: Configurable boss appearance
- 📦 **Package Settings**: After boss drops, chance rates
- ⚔️ **Battle Condition**: Cooldown reduction bonus
- 🌌 **Galaxy Compressor**: Package cooldown reduction (Epic to Ancestral)
- 🌐 **Multiverse Nexus**: Synchronize BH/GT/DW cooldowns (Epic to Ancestral)
- 🌾 **Farming Perks**: Extended durations for farming mode

### Analysis Tools
- 📊 **Real-time Simulation**: 1-hour timeline simulation
- 📈 **Interactive Charts**: Visual uptime/downtime analysis
- ✨ **Package Visualization**: See exactly when packages drop
- 🎯 **Permanence Detection**: Automatically identify permanent uptime
- 📉 **Statistics Cards**: Detailed metrics per Ultimate Weapon
- 🔄 **Synchronized View**: Compare all UWs on one chart

## Technology Stack

- **Framework**: Dash (Python web framework)
- **Visualization**: Plotly
- **UI**: Dash Bootstrap Components (DARKLY theme)
- **Computation**: NumPy, Pandas
- **Deployment**: Render.com (free tier)

## Local Development

### Requirements
- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/uw-permacalc.git
cd uw-permacalc

# Install dependencies
pip install -r requirements.txt

# Run the app
python perma_calc_new.py
```

Open your browser to: **http://127.0.0.1:8050**

## Deployment

This app is configured for easy deployment to Render.com:

1. Fork/clone this repo
2. Create a new Web Service on Render.com
3. Connect your GitHub repository
4. Render will auto-detect settings from configuration

**Build Command**: `pip install -r requirements.txt`  
**Start Command**: `gunicorn perma_calc_new:server`

See [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md) for detailed instructions.

## How It Works

The calculator simulates game mechanics second-by-second:
1. **Cooldown System**: Tracks UW cooldowns with package reductions
2. **Activation Logic**: Handles overlapping activations and stacking
3. **Package Mechanics**: Simulates random package drops and boss packages
4. **Synchronization**: MVN effect synchronizes multiple UWs
5. **Perks & Bonuses**: Applies all modifiers (BC, farming perks, etc.)

## Game Mechanics

### Cooldown Reduction
- **Packages**: Random cooldown reduction (13-20s based on Galaxy Compressor)
- **Boss Packages**: Guaranteed packages after boss waves
- **Battle Condition**: +10s to all cooldowns (reduces available uptime)

### Wave Accelerator
Reduces wave time from base 26s:
- **Tournament**: Base wave time reduced by card level
- **Farming**: Longer waves, more package opportunities

### Multiverse Nexus
Synchronizes Black Hole, Golden Tower, and Death Wave to activate together:
- Calculates average cooldown of the three
- Applies tier offset (Epic: +20s, Legendary: +10s, Mythic: +1s, Ancestral: -10s)
- All three UWs activate simultaneously

## Contributing

This is a personal project, but suggestions and bug reports are welcome via GitHub Issues.

## License

MIT License - feel free to use and modify for your own purposes.

## Credits

Created for The Tower game community.

**Game**: [The Tower](https://www.thetower.lol/)  
**Tools Reference**: [MVN Tools](https://mvn.thetower.tools/)

---

## Screenshots

*(Add screenshots after deployment)*

- Statistics Overview
- Detailed UW Analysis Chart
- Synchronized Timeline View
- Configuration Panel

---

**Last Updated**: January 2026
