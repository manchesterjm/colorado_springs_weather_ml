# Colorado Springs Weather ML

Daily ML weather forecasting for Colorado Springs (KCOS). Generates h=1..7 day forecasts at 06:30 MST every morning, compares to NWS, blends, logs every forecast for verification, and retrains nightly.

**Reference doc:** `D:\Documents\Claude_References\Production_Weather_ML_Reference.md` — full architecture, schema, code modules, runbook, troubleshooting.

**Sister project:** `D:\Projects\Markov_Weather_Forecasting\` (research artifact / paper). This project imports `D:\Scripts\weather_regime_pkg` as a library.

---

## Quick Start

### View latest forecast
```powershell
Get-ChildItem D:\Projects\CO_Springs_Weather_ML\reports\*.md | Sort-Object LastWriteTime | Select-Object -Last 1 | Get-Content
```

### Run a forecast on demand
```powershell
python D:\Projects\CO_Springs_Weather_ML\scripts\forecast_runner.py
# or print without writing to DB:
python D:\Projects\CO_Springs_Weather_ML\scripts\forecast_runner.py --print-only
```

### View accuracy dashboard
```powershell
python D:\Projects\CO_Springs_Weather_ML\scripts\accuracy_dashboard.py
python D:\Projects\CO_Springs_Weather_ML\scripts\accuracy_dashboard.py --windows 7,14,30
python D:\Projects\CO_Springs_Weather_ML\scripts\accuracy_dashboard.py --markdown
```

### Verify scheduled tasks are registered
```powershell
Get-ScheduledTask -TaskName "Forecast ML*" | Format-Table TaskName, State, LastRunTime
```

### Re-register tasks (after pulling code changes; run as Admin)
```powershell
D:\Projects\CO_Springs_Weather_ML\scripts\Create-MLLoggerTasks.ps1
```

---

## Scheduled Tasks (5)

| Task                         | When (MST) | Script                   | Purpose                                    |
| ---------------------------- | ---------- | ------------------------ | ------------------------------------------ |
| Forecast ML Hourly METAR     | every :05  | `metar_ml_logger.py`     | Pull NWS observations for 23 stations      |
| Forecast ML Daily Aggregator | 00:30      | `daily_aggregator.py`    | Aggregate `metar_hourly` → `daily_summary` |
| Forecast ML Retrain          | 02:00      | `retrain.py`             | Refit all 28 models in production mode     |
| Forecast ML Daily Run        | 06:30      | `forecast_runner.py`     | Issue daily 7-day forecast, blend, render  |
| Forecast ML Verification     | 23:55      | `verification_logger.py` | Score forecasts vs actuals                 |

---

## Architecture

```
forecast_ml.db (this project)        weather.db (existing, READ-ONLY here)
├── metar_hourly      (6.3M rows)    ├── ambient_observations  (PWS)
├── daily_summary     (202K rows)    ├── forecast_snapshots    (NWS text)
├── production_forecast              ├── digital_forecast      (NWS digital)
├── forecast_verification            └── actual_daily_climate
├── model_runs
├── training_cache
├── nws_compare_cache
└── backfill_progress
```

Reads `weather_regime_pkg` from `D:\Scripts\` for the base regime classifier (state classification, base features, NWS blend, training scaffolding). Adds Hi/Lo regressors + PoP classifier on the same feature matrix.

---

## Layout

```
CO_Springs_Weather_ML\
├── forecast_ml_pkg\    library: db schema, stations, features, models
├── scripts\            entry points run by Task Scheduler
├── data\               forecast_ml.db (~2.2 GB), caches, logs, model_runs/*.pkl
├── reports\            daily markdown reports
└── tests\              (scaffolded, empty)
```

---

## Stations (23)

| Group          | Stations                                 |
| -------------- | ---------------------------------------- |
| Local / Plains | KCOS, KFLY, KAFF, KFCS, KAPA, KPUB       |
| Adjacent       | KDEN, KCYS                               |
| Mid-W mountain | KLXV, KAEJ, KASE, KEGE, KGUC, KGJT, KMTJ |
| SW synoptic    | KDRO, KFMN, KFLG                         |
| Far-W synoptic | KSLC, KBOI                               |
| S synoptic     | KELP, KAMA, KDDC                         |

---

## Models

The active model is identified by `model_runs.is_active = 1`. Retraining each night demotes the previous active row to `is_active = 0` and inserts a new active row pointing to a new pickle file.

Each artifact contains 28 models:
- 7 horizons × {regime, hi, lo, pop}
- regime + pop heads are isotonic-calibrated CalibratedClassifierCV
- hi/lo heads are HistGradientBoostingRegressor (squared_error)

---

## Phase History

All 5 phases shipped 2026-05-09 (one session). See the reference doc for full per-phase details.

| Phase     | Theme                          | LoC        |
| --------- | ------------------------------ | ---------: |
| 1         | Data Infrastructure            | 1,318      |
| 2         | Real-time Ingestion            | 330        |
| 3         | Feature Engineering & Training | 560        |
| 4         | Production Runner              | 665        |
| 5         | Verification & Retrain         | 530        |
| **Total** |                                | **~3,400** |

---

## Plan

`~/.claude/plans/mutable-imagining-stonebraker.md` (WSL home).
