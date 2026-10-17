# Assistive Vision

A real-time navigation aid for blind and low-vision people. A chest-mounted phone camera detects and tracks hazards on pavements and warns the user through audio and haptic feedback.

> **Status:** early development (Phase 0). Not a safety device. It is designed to be used **alongside** a white cane or guide dog, never as a replacement.

## Focus

The aid targets what a cane and hearing miss, rather than alerting on everything:

1. Head- and chest-height obstacles (branches, wing mirrors, signs, scaffolding)
2. Quiet, fast movers on the pavement (cyclists, e-scooters, vehicles crossing at driveways)
3. Blocked pavements ahead (parked vehicles, bins, A-boards)

Every object is tracked in the background, but the user is only alerted when something matters.

## Project structure

```
data/        datasets (never committed; downloaded per session)
scripts/     one-off utilities (dataset audit, label review)
src/         project code as an importable package
notebooks/   experiments that import from src/
```

## Setup (local)

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Training runs on Google Colab GPU runtimes.

## Data and attribution

This project uses the **Sidewalk Obstacle Detection (SOD)** dataset, version 1, by LaMAO on Roboflow Universe, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Source: <https://universe.roboflow.com/lamao/sod-enect/dataset/1>

> Boussihmed, A., El Makkaoui, K., Ouahbi, I., Maleh, Y., & Chetouani, A. (2025). A TinyML model for sidewalk obstacle detection: aiding the blind and visually impaired people. *Multimedia Tools and Applications*, 84(22), 25837–25864.

Only a subset of classes is used, and labels may be remapped.
