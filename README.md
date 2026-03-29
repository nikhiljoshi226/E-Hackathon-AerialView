# E-Hackathon Aerial View

This repository contains our final working notebooks, generated outputs, and supporting data for all three hackathon problem statements.

## Repository Structure

```text
E-Hackathon Aerial View/
├── README.md
├── data_dir/
│   ├── equity/
│   │   ├── market_data.csv
│   │   ├── ohlcv.csv
│   │   └── trade_data.csv
│   ├── crypto-market/
│   │   ├── Binance_BATUSDT_2026_minute.csv
│   │   ├── Binance_BTCUSDT_2026_minute.csv
│   │   ├── Binance_DOGEUSDT_2026_minute.csv
│   │   ├── Binance_ETHUSDT_2026_minute.csv
│   │   ├── Binance_LTCUSDT_2026_minute.csv
│   │   ├── Binance_SOLUSDT_2026_minute.csv
│   │   ├── Binance_USDCUSDT_2026_minute.csv
│   │   └── Binance_XRPUSDT_2026_minute.csv
│   └── crypto-trades/
│       ├── BATUSDT_trades.csv
│       ├── BTCUSDT_trades.csv
│       ├── DOGEUSDT_trades.csv
│       ├── ETHUSDT_trades.csv
│       ├── LTCUSDT_trades.csv
│       ├── SOLUSDT_trades.csv
│       ├── USDCUSDT_trades.csv
│       └── XRPUSDT_trades.csv
├── src/
│   ├── ProblemStatement1.ipynb
│   ├── ProblemStatement2.ipynb
│   ├── ProblemStatement3.ipynb
│   ├── p1_alerts.csv
│   ├── p2_signals.csv
│   └── submission.csv
└── outputs/
    ├── ProblemStatement1_o1.png
    ├── ProblemStatement2_o1.png
    ├── ProblemStatement2_o2.png
    ├── ProblemStatement3_o1.png
    ├── ProblemStatement3_o2.png
    ├── ProblemStatement3_o3.png
    ├── ProblemStatement3_o4.png
    └── ProblemStatement3_o5.png
```

## Environment

Recommended Python version: **3.10+**

Install required libraries before running any notebook:

```bash
pip install pandas numpy scikit-learn requests beautifulsoup4 matplotlib jupyter
```

## Problem 1 — Order Book Concentration (Equity)

Notebook:
- `src/ProblemStatement1.ipynb`

Inputs used:
- `data_dir/equity/market_data.csv`
- `data_dir/equity/ohlcv.csv`
- `data_dir/equity/trade_data.csv`

Output generated:
- `src/p1_alerts.csv`

What this notebook does:
- loads per-minute equity order book, OHLCV, and trade data
- computes order book imbalance, spread in basis points, level-1 concentration, depth ratio, and rolling z-scores
- detects suspicious windows such as sustained imbalance, spread/depth dislocation, and other abnormal order book behavior
- compresses overlapping candidate windows into stronger episode-level alerts
- writes a structured alert file with required columns
- prints runtime and summary logs for screenshot-ready output

Expected final file format:
- `alert_id`
- `sec_id`
- `trade_date`
- `time_window_start`
- `anomaly_type`
- `severity`
- `remarks`
- `time_to_run`

## Problem 2 — Insider Trading Signal (Equity)

Notebook:
- `src/ProblemStatement2.ipynb`

Inputs used:
- `data_dir/equity/ohlcv.csv`
- `data_dir/equity/trade_data.csv`
- SEC EDGAR public 8-K search results fetched by the notebook

Output generated:
- `src/p2_signals.csv`

What this notebook does:
- builds an EDGAR 8-K event timeline
- classifies material events using headline keyword matching
- computes 15-day rolling baselines for volume and returns
- checks for pre-announcement drift and abnormal volume before filing dates
- inspects trade-level activity in the suspicious window
- writes a structured signal file with required columns
- prints runtime and summary logs for screenshot-ready output

Expected final file format:
- `sec_id`
- `event_date`
- `event_type`
- `headline`
- `source_url`
- `pre_drift_flag`
- `suspicious_window_start`
- `remarks`
- `time_to_run`

## Problem 3 — Crypto Blind Anomaly Hunt (Compulsory)

Notebook:
- `src/ProblemStatement3.ipynb`

Inputs used:
- `data_dir/crypto-market/*.csv`
- `data_dir/crypto-trades/*.csv`

Output generated:
- `src/submission.csv`

What this notebook does:
- loads all 8 crypto market and trade files
- uses minute-level market data as baseline context for trade anomalies
- looks for suspicious trade behavior such as structuring, wash-style reversals, ramping patterns, peg-related events, and manager-linked consolidation behavior
- keeps `violation_type` aligned to the accepted taxonomy
- writes one row per suspicious trade ID
- prints runtime, per-symbol counts, and final preview logs for screenshots

Accepted violation types used in the project follow the official taxonomy, such as:
- `aml_structuring`
- `coordinated_structuring`
- `threshold_testing`
- `chain_layering`
- `manager_consolidation`
- `placement_smurfing`
- `wash_trading`
- `pump_and_dump`
- `layering_echo`
- `spoofing`
- `ramping`
- `coordinated_pump`
- `round_trip_wash`
- `cross_pair_divergence`
- `peg_break`
- `wash_volume_at_peg`

Expected final file format:
- `symbol`
- `date`
- `trade_id`
- `violation_type`
- `remarks` *(optional but included for reviewer clarity)*

## How to Run

Open Jupyter in the repository root and run the notebooks in `src/` one by one.

```bash
jupyter notebook
```

Recommended execution order:
1. `src/ProblemStatement1.ipynb`
2. `src/ProblemStatement2.ipynb`
3. `src/ProblemStatement3.ipynb`

Each notebook:
- reads its required files from `data_dir/`
- prints timing and summary logs
- writes the final CSV into `src/`

## Outputs Folder

The `outputs/` folder contains screenshot-ready images captured from notebook runs. These are useful for presentation, documentation, or hackathon submission support.

## Submission Checklist

Before final submission, verify the following:

- `src/p1_alerts.csv` exists and opens correctly
- `src/p2_signals.csv` exists and opens correctly
- `src/submission.csv` exists and opens correctly
- all three notebooks run from top to bottom without path changes
- runtime logs are visible in notebook output
- remarks columns are populated and readable
- `submission.csv` is the final Problem 3 file being submitted
- this `README.md` is included at repo root

## Notes

- Problem 3 is the highest-value problem, so false positives matter a lot. The current notebook is designed to balance coverage with defensible trade selection.
- Problem 1 and Problem 2 are bonus problems, but both outputs are included in final form.
- The notebooks are written to be clean for screenshots and final review, with reduced clutter in printed outputs.
