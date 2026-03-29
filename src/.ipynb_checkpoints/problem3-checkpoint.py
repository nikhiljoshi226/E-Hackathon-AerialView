# =============================================================
# PROBLEM 3 — CRYPTO BLIND ANOMALY HUNT (450 pts)
# =============================================================
# HOW TO RUN:
#   python problem3.py
#
# WHAT IT DOES:
#   Loads all 8 crypto pair files, runs detection logic for each,
#   and writes submission.csv at the end.
#
# BEFORE RUNNING:
#   1. Put all your SYMBOL_market.csv and SYMBOL_trades.csv files
#      in the same folder as this script (or set DATA_DIR below)
#   2. pip install pandas numpy scikit-learn
# =============================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import time
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────
# Change DATA_DIR to wherever your CSV files live
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data_dir")

# All 8 pairs — ordered easiest → hardest for maximum early points
PAIRS = [
    "USDCUSDT",   # ← Start here. Any price ≠ $1.00 is suspicious
    "BATUSDT",    # ← Very illiquid. Any activity stands out
    "DOGEUSDT",   # ← Pump-and-dump signatures
    "LTCUSDT",    # ← Wash trading and round trips
    "XRPUSDT",    # ← AML structuring patterns
    "SOLUSDT",    # ← Isolation Forest works well here
    "ETHUSDT",    # ← Needs tighter feature engineering
    "BTCUSDT",    # ← Hardest. Do last.
]

start_time = time.time()
# ─────────────────────────────────────────────────────────────


# =============================================================
# STEP 1 — LOAD DATA
# =============================================================

def load_pair_data(symbol):
    """
    Loads market and trades CSV files for a given symbol.
    Adds a 'date' column (YYYY-MM-DD string) to both DataFrames.
    """
    market_path = os.path.join(DATA_DIR, "crypto-market", f"Binance_{symbol}_2026_minute.csv")
    trades_path = os.path.join(DATA_DIR, "crypto-trades", f"{symbol}_trades.csv")

    market = pd.read_csv(market_path, parse_dates=["timestamp"])
    trades = pd.read_csv(trades_path, parse_dates=["timestamp"])

    # Add clean date column needed for submission
    trades["date"] = trades["timestamp"].dt.strftime("%Y-%m-%d")
    market["date"] = market["timestamp"].dt.strftime("%Y-%m-%d")

    # Compute notional (price × quantity) — used by many detectors
    trades["notional"] = trades["price"] * trades["quantity"]

    print(f"\n  {symbol}: {len(trades):,} trades | "
          f"price {trades['price'].min():.4f}–{trades['price'].max():.4f} | "
          f"qty {trades['quantity'].min():.2f}–{trades['quantity'].max():.2f}")

    return market, trades


# =============================================================
# STEP 2 — DETECTION FUNCTIONS (one per pattern type)
# =============================================================

# ──────────────────────────────────────────────────────────────
# DETECTOR 1: PEG BREAK (USDCUSDT only)
# USDC should always be exactly $1.00. Any deviation is a flag.
# Violation types: peg_break, wash_volume_at_peg
# ──────────────────────────────────────────────────────────────
def detect_peg_break(trades, market, symbol):
    results = []

    # ── peg_break: price deviates more than 0.5% from $1.00 ──
    peg_broken = trades[abs(trades["price"] - 1.0) > 0.005].copy()

    for _, row in peg_broken.iterrows():
        dev_pct = abs(row["price"] - 1.0) * 100
        results.append({
            "symbol":         symbol,
            "date":           row["date"],
            "trade_id":       row["trade_id"],
            "violation_type": "peg_break",
            "remarks": (
                f"Price {row['price']:.5f} deviates {dev_pct:.3f}% from $1.00 stablecoin peg. "
                f"Quantity {row['quantity']:.2f}. "
                f"USDC is always $1.00 by design — any deviation with real volume is a manipulation signal."
            )
        })

    # ── wash_volume_at_peg: same wallet buys and sells at $1.00
    #    with near-zero net position (fake volume creation) ────
    if "wallet_id" in trades.columns:
        at_peg = trades[abs(trades["price"] - 1.0) <= 0.001].copy()
        for wallet, wt in at_peg.groupby("wallet_id"):
            if len(wt) < 4:
                continue
            buy_qty  = wt[wt["side"] == "BUY"]["quantity"].sum()
            sell_qty = wt[wt["side"] == "SELL"]["quantity"].sum()
            total    = buy_qty + sell_qty
            if total == 0:
                continue
            net_ratio = abs(buy_qty - sell_qty) / total
            if net_ratio < 0.05:          # Less than 5% net = wash
                for _, row in wt.iterrows():
                    results.append({
                        "symbol":         symbol,
                        "date":           row["date"],
                        "trade_id":       row["trade_id"],
                        "violation_type": "wash_volume_at_peg",
                        "remarks": (
                            f"Wallet {wallet} has {len(wt)} trades at $1.00 peg. "
                            f"Buy qty={buy_qty:.2f}, Sell qty={sell_qty:.2f}, "
                            f"Net ratio={net_ratio:.4f}. "
                            f"Near-zero net position = artificial volume inflation at peg."
                        )
                    })

    print(f"    → peg_break / wash_volume_at_peg: {len(results)} flags")
    return results


# ──────────────────────────────────────────────────────────────
# DETECTOR 2: AML STRUCTURING / SMURFING (BATUSDT, XRPUSDT, LTCUSDT)
# Many trades just below a round-number threshold to avoid reporting.
# Violation types: aml_structuring, coordinated_structuring, threshold_testing
# ──────────────────────────────────────────────────────────────
def detect_aml_structuring(trades, market, symbol):
    results = []

    # Common reporting thresholds (in USDT notional value)
    THRESHOLDS   = [1_000, 5_000, 10_000, 50_000, 100_000]
    BUFFER       = 0.05    # Flag trades within 5% BELOW threshold
    MIN_TRADES   = 3       # Need at least this many to call it structuring
    MAX_CV       = 0.05    # Coefficient of variation must be < 5% (very uniform sizes)

    structuring_wallets = {}   # Track which wallets are structuring around which threshold

    if "wallet_id" not in trades.columns:
        return results

    for wallet, wt in trades.groupby("wallet_id"):
        if len(wt) < MIN_TRADES:
            continue

        for thresh in THRESHOLDS:
            lower  = thresh * (1 - BUFFER)
            bucket = wt[(wt["notional"] >= lower) & (wt["notional"] < thresh)]

            if len(bucket) < MIN_TRADES:
                continue

            mean_n = bucket["notional"].mean()
            std_n  = bucket["notional"].std()
            cv     = (std_n / mean_n) if mean_n > 0 else 999

            if cv < MAX_CV:      # Deliberately uniform = structuring
                structuring_wallets.setdefault(thresh, []).append(wallet)
                for _, row in bucket.iterrows():
                    results.append({
                        "symbol":         symbol,
                        "date":           row["date"],
                        "trade_id":       row["trade_id"],
                        "violation_type": "aml_structuring",
                        "remarks": (
                            f"Wallet {wallet}: {len(bucket)} trades, "
                            f"avg notional {mean_n:.2f} USDT (CV={cv:.4f}), "
                            f"all just below {thresh} threshold. "
                            f"Deliberately uniform sizes to avoid reporting — classic smurfing."
                        )
                    })

    # ── coordinated_structuring: multiple wallets, same threshold, same day ──
    for thresh, wallets in structuring_wallets.items():
        if len(wallets) < 2:
            continue
        involved_trades = trades[
            (trades["wallet_id"].isin(wallets)) &
            (trades["notional"] >= thresh * (1 - BUFFER)) &
            (trades["notional"] < thresh)
        ]
        dates_with_multiple = (
            involved_trades.groupby("date")["wallet_id"]
            .nunique()
        )
        coordinated_dates = set(dates_with_multiple[dates_with_multiple >= 2].index)

        for i, res in enumerate(results):
            if res["symbol"] == symbol and res["date"] in coordinated_dates:
                matching = trades[trades["trade_id"] == res["trade_id"]]
                if not matching.empty and matching.iloc[0]["wallet_id"] in wallets:
                    results[i]["violation_type"] = "coordinated_structuring"
                    results[i]["remarks"] += (
                        f" COORDINATED: wallets {wallets} all structured "
                        f"below {thresh} on {res['date']}."
                    )

    # ── threshold_testing: one trade AT the limit, then many below it ──
    for wallet, wt in trades.groupby("wallet_id"):
        for thresh in THRESHOLDS:
            at     = wt[(wt["notional"] >= thresh * 0.999) & (wt["notional"] <= thresh * 1.001)]
            below  = wt[wt["notional"] < thresh * 0.99]
            if len(at) >= 1 and len(below) >= 3:
                for _, row in at.iterrows():
                    results.append({
                        "symbol":         symbol,
                        "date":           row["date"],
                        "trade_id":       row["trade_id"],
                        "violation_type": "threshold_testing",
                        "remarks": (
                            f"Wallet {wallet} placed trade at exactly "
                            f"{row['notional']:.2f} USDT (threshold={thresh}), "
                            f"then followed with {len(below)} trades below threshold. "
                            f"Testing the limit before starting a structuring campaign."
                        )
                    })

    # Deduplicate
    results = _dedup(results)
    print(f"    → aml_structuring / coordinated / threshold_testing: {len(results)} flags")
    return results


# ──────────────────────────────────────────────────────────────
# DETECTOR 3: PUMP AND DUMP (DOGEUSDT, SOLUSDT)
# Rising price + rising volume over multiple bars → sharp crash.
# Violation types: pump_and_dump, coordinated_pump
# ──────────────────────────────────────────────────────────────
def detect_pump_and_dump(trades, market, symbol):
    results  = []
    market   = market.copy().sort_values("timestamp").reset_index(drop=True)

    # Compute rolling baselines
    market["price_change"] = market["close"].pct_change()
    market["vol_mean20"]   = market["volume"].rolling(20, min_periods=1).mean()
    market["vol_std20"]    = market["volume"].rolling(20, min_periods=1).std().fillna(1)
    market["vol_z"]        = (market["volume"] - market["vol_mean20"]) / market["vol_std20"].replace(0, 1)

    PUMP_BARS  = 5             # Need at least 5 consecutive rising bars
    DUMP_DROP  = -0.015        # Dump = price falls >1.5% in 1–2 bars
    VOL_MULT   = 2.0           # Volume must be 2× normal during pump
    COORD_WALLETS = 3          # 3+ wallets = coordinated_pump

    i = 0
    while i < len(market) - PUMP_BARS - 2:
        window = market.iloc[i : i + PUMP_BARS]

        # Check consecutive price rises
        prices_rising = all(
            window["close"].iloc[j] < window["close"].iloc[j + 1]
            for j in range(len(window) - 1)
        )

        # Check elevated volume
        high_volume = (
            window["vol_z"].mean() > VOL_MULT or
            window["volume"].mean() > market["volume"].mean() * VOL_MULT
        )

        if prices_rising and high_volume:
            # Check for dump in the next 2 bars
            dump_window = market.iloc[i + PUMP_BARS : i + PUMP_BARS + 2]
            if not dump_window.empty and dump_window["price_change"].min() < DUMP_DROP:
                pump_start = window["timestamp"].iloc[0]
                pump_end   = window["timestamp"].iloc[-1]
                dump_bar   = dump_window["timestamp"].iloc[0]

                # Trades during pump = BUY side suspects
                pump_buys = trades[
                    (trades["timestamp"] >= pump_start) &
                    (trades["timestamp"] <= dump_bar) &
                    (trades["side"] == "BUY")
                ]

                # Trades during dump = SELL side suspects
                dump_sells = trades[
                    (trades["timestamp"] >= pump_end) &
                    (trades["timestamp"] <= dump_bar + pd.Timedelta(minutes=2)) &
                    (trades["side"] == "SELL")
                ]

                n_wallets  = pump_buys["wallet_id"].nunique() if "wallet_id" in pump_buys.columns else 0
                vtype      = "coordinated_pump" if n_wallets >= COORD_WALLETS else "pump_and_dump"
                price_gain = (window["close"].iloc[-1] / window["close"].iloc[0] - 1) * 100
                max_drop   = dump_window["price_change"].min() * 100

                for _, row in pd.concat([pump_buys, dump_sells]).iterrows():
                    side_label = "pump BUY" if row["side"] == "BUY" else "dump SELL"
                    results.append({
                        "symbol":         symbol,
                        "date":           row["date"],
                        "trade_id":       row["trade_id"],
                        "violation_type": vtype,
                        "remarks": (
                            f"{side_label} during {PUMP_BARS}-bar pump-and-dump. "
                            f"Pump: price rose {price_gain:.1f}% "
                            f"({window['close'].iloc[0]:.4f}→{window['close'].iloc[-1]:.4f}). "
                            f"Dump: price fell {abs(max_drop):.1f}% in 1–2 bars. "
                            f"{n_wallets} wallets bought during pump phase."
                        )
                    })
                i += PUMP_BARS
                continue
        i += 1

    results = _dedup(results)
    print(f"    → pump_and_dump / coordinated_pump: {len(results)} flags")
    return results


# ──────────────────────────────────────────────────────────────
# DETECTOR 4: WASH TRADING & ROUND TRIP WASH (LTCUSDT, BATUSDT, BTC, ETH)
# Same wallet or paired wallets buying/selling with near-zero net position.
# Violation types: wash_trading, round_trip_wash
# ──────────────────────────────────────────────────────────────
def detect_wash_trading(trades, market, symbol):
    results = []

    if "wallet_id" not in trades.columns:
        return results

    NET_RATIO_THRESH  = 0.10    # Less than 10% net position = wash
    MIN_TRADES_WALLET = 4       # Need at least 4 trades to be meaningful
    PRICE_TOL         = 0.002   # 0.2% price tolerance for pair matching
    QTY_TOL           = 0.02    # 2% quantity tolerance for pair matching

    # ── wash_trading: single wallet buys and sells, near-zero net ──
    for wallet, wt in trades.groupby("wallet_id"):
        if len(wt) < MIN_TRADES_WALLET:
            continue

        buy_qty  = wt[wt["side"] == "BUY"]["quantity"].sum()
        sell_qty = wt[wt["side"] == "SELL"]["quantity"].sum()
        total    = buy_qty + sell_qty
        if total == 0:
            continue

        net_ratio = abs(buy_qty - sell_qty) / total
        if net_ratio < NET_RATIO_THRESH:
            for _, row in wt.iterrows():
                results.append({
                    "symbol":         symbol,
                    "date":           row["date"],
                    "trade_id":       row["trade_id"],
                    "violation_type": "wash_trading",
                    "remarks": (
                        f"Wallet {wallet}: {len(wt)} trades, "
                        f"buy_qty={buy_qty:.4f}, sell_qty={sell_qty:.4f}, "
                        f"net_ratio={net_ratio:.4f}. "
                        f"Near-zero net position — no genuine ownership change. Wash trading."
                    )
                })

    # ── round_trip_wash: two DIFFERENT wallets mirror each other ──
    trades_sorted  = trades.sort_values("timestamp").copy()
    trades_sorted["time_bucket"] = trades_sorted["timestamp"].dt.floor("5min")

    seen_pairs = set()
    for tb, bucket in trades_sorted.groupby("time_bucket"):
        if len(bucket) < 4:
            continue

        buys  = bucket[bucket["side"] == "BUY"]
        sells = bucket[bucket["side"] == "SELL"]

        for _, buy_row in buys.iterrows():
            for _, sell_row in sells.iterrows():
                w_buy  = buy_row.get("wallet_id",  "")
                w_sell = sell_row.get("wallet_id", "")
                if w_buy == w_sell or w_buy == "" or w_sell == "":
                    continue

                pair_key = tuple(sorted([buy_row["trade_id"], sell_row["trade_id"]]))
                if pair_key in seen_pairs:
                    continue

                qty_diff   = abs(buy_row["quantity"] - sell_row["quantity"]) / max(buy_row["quantity"], 1e-9)
                price_diff = abs(buy_row["price"]    - sell_row["price"])    / max(buy_row["price"],    1e-9)

                if qty_diff < QTY_TOL and price_diff < PRICE_TOL:
                    seen_pairs.add(pair_key)
                    for trade_id, wallet, row in [
                        (buy_row["trade_id"],  w_buy,  buy_row),
                        (sell_row["trade_id"], w_sell, sell_row),
                    ]:
                        if not any(r["trade_id"] == trade_id for r in results):
                            results.append({
                                "symbol":         symbol,
                                "date":           row["date"],
                                "trade_id":       trade_id,
                                "violation_type": "round_trip_wash",
                                "remarks": (
                                    f"Wallet {w_buy} bought {buy_row['quantity']:.4f} "
                                    f"while wallet {w_sell} sold {sell_row['quantity']:.4f} "
                                    f"at nearly identical price {buy_row['price']:.4f}. "
                                    f"Quantities match within {qty_diff*100:.2f}%, "
                                    f"prices within {price_diff*100:.2f}% — round-trip wash trade."
                                )
                            })

    results = _dedup(results)
    print(f"    → wash_trading / round_trip_wash: {len(results)} flags")
    return results


# ──────────────────────────────────────────────────────────────
# DETECTOR 5: RAMPING (LTCUSDT, XRPUSDT)
# Same wallet makes sequential BUY trades at monotonically rising prices.
# Violation type: ramping
# ──────────────────────────────────────────────────────────────
def detect_ramping(trades, market, symbol):
    results = []

    if "wallet_id" not in trades.columns:
        return results

    MIN_TRADES    = 5      # Minimum trades in a ramping sequence
    DIR_THRESHOLD = 0.70   # 70%+ of price moves must be in same direction
    BUY_THRESHOLD = 0.70   # 70%+ must be BUY orders

    for wallet, wt in trades.groupby("wallet_id"):
        if len(wt) < MIN_TRADES:
            continue

        wt = wt.sort_values("timestamp").reset_index(drop=True)

        # Count how many consecutive price changes are upward
        price_changes = wt["price"].diff().dropna()
        rising_count  = (price_changes > 0).sum()
        total_changes = len(price_changes)

        buy_ratio = (wt["side"] == "BUY").mean()

        if (rising_count / total_changes >= DIR_THRESHOLD and
                len(wt) >= MIN_TRADES and
                buy_ratio >= BUY_THRESHOLD):

            price_start = wt["price"].iloc[0]
            price_end   = wt["price"].iloc[-1]
            price_move  = (price_end / price_start - 1) * 100

            for _, row in wt.iterrows():
                results.append({
                    "symbol":         symbol,
                    "date":           row["date"],
                    "trade_id":       row["trade_id"],
                    "violation_type": "ramping",
                    "remarks": (
                        f"Wallet {wallet}: {len(wt)} trades, "
                        f"{rising_count/total_changes*100:.0f}% price moves upward, "
                        f"price walked from {price_start:.4f} to {price_end:.4f} "
                        f"(+{price_move:.2f}%), {buy_ratio*100:.0f}% BUY side. "
                        f"Systematically advancing price — ramping pattern."
                    )
                })

    results = _dedup(results)
    print(f"    → ramping: {len(results)} flags")
    return results


# ──────────────────────────────────────────────────────────────
# DETECTOR 6: LAYERING ECHO (ETHUSDT, BTCUSDT)
# Wallet places many trades in one direction then immediately reverses.
# Violation type: layering_echo
# ──────────────────────────────────────────────────────────────
def detect_layering_echo(trades, market, symbol):
    results = []

    if "wallet_id" not in trades.columns:
        return results

    TIME_WINDOW         = pd.Timedelta(minutes=15)
    MIN_PER_SIDE        = 3       # Need at least 3 buys + 3 sells to call it layering

    for wallet, wt in trades.groupby("wallet_id"):
        if len(wt) < MIN_PER_SIDE * 2:
            continue

        wt = wt.sort_values("timestamp").reset_index(drop=True)

        for i in range(len(wt) - MIN_PER_SIDE * 2 + 1):
            window = wt.iloc[i : i + MIN_PER_SIDE * 2 + 2]
            duration = window["timestamp"].max() - window["timestamp"].min()

            if duration > TIME_WINDOW:
                continue

            sides = window["side"].tolist()
            first_half  = sides[:MIN_PER_SIDE]
            second_half = sides[MIN_PER_SIDE:]

            is_buy_then_sell  = all(s == "BUY"  for s in first_half)  and all(s == "SELL" for s in second_half)
            is_sell_then_buy  = all(s == "SELL" for s in first_half)  and all(s == "BUY"  for s in second_half)

            if is_buy_then_sell or is_sell_then_buy:
                direction = "BUY→SELL" if is_buy_then_sell else "SELL→BUY"
                for _, row in window.iterrows():
                    results.append({
                        "symbol":         symbol,
                        "date":           row["date"],
                        "trade_id":       row["trade_id"],
                        "violation_type": "layering_echo",
                        "remarks": (
                            f"Wallet {wallet}: {direction} pattern — "
                            f"{MIN_PER_SIDE}+ trades in one direction then immediately reversed, "
                            f"all within {duration.seconds//60} minutes. "
                            f"Classic layering echo: push price in one direction then exit."
                        )
                    })
                break   # One layering event per wallet per scan

    results = _dedup(results)
    print(f"    → layering_echo: {len(results)} flags")
    return results


# ──────────────────────────────────────────────────────────────
# DETECTOR 7: ISOLATION FOREST (SOLUSDT, ETHUSDT, BTCUSDT)
# ML-based anomaly detector. Catches patterns the rule-based
# detectors miss by looking at multiple features at once.
# ──────────────────────────────────────────────────────────────
def detect_isolation_forest(trades, market, symbol):
    results = []

    if len(trades) < 50:
        return results

    trades  = trades.copy()
    market  = market.copy()

    # ── Feature 1: Quantity z-score (rolling 20-bar) ──
    trades["qty_roll_mean"] = trades["quantity"].rolling(20, min_periods=1).mean()
    trades["qty_roll_std"]  = trades["quantity"].rolling(20, min_periods=1).std().fillna(1).replace(0, 1)
    trades["qty_z"]         = (trades["quantity"] - trades["qty_roll_mean"]) / trades["qty_roll_std"]

    # ── Feature 2: Price deviation from 1-minute bar mid ──
    trades["minute"]  = trades["timestamp"].dt.floor("1min")
    market["minute"]  = market["timestamp"].dt.floor("1min")
    market["bar_mid"] = (market["high"] + market["low"]) / 2

    trades = trades.merge(market[["minute", "bar_mid"]], on="minute", how="left")
    trades["price_dev"] = (
        (trades["price"] - trades["bar_mid"].fillna(trades["price"])).abs() /
        trades["bar_mid"].fillna(trades["price"]).replace(0, 1)
    )

    # ── Feature 3: Wallet trade frequency ──
    if "wallet_id" in trades.columns:
        freq_map = trades.groupby("wallet_id")["trade_id"].count()
        trades["wallet_freq"] = trades["wallet_id"].map(freq_map).fillna(1)
    else:
        trades["wallet_freq"] = 1

    # ── Feature 4: Hour of day (end-of-period trades are more suspicious) ──
    trades["hour"] = trades["timestamp"].dt.hour

    # ── Run Isolation Forest ──
    feature_cols = ["qty_z", "price_dev", "wallet_freq", "hour"]
    feat_data    = trades[feature_cols].fillna(0)

    scaler = StandardScaler()
    X      = scaler.fit_transform(feat_data)

    clf    = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
    preds  = clf.fit_predict(X)
    scores = clf.score_samples(X)

    trades["anomaly_score"] = scores
    trades["is_anomaly"]    = (preds == -1)

    # Only flag the most extreme 2% of anomalies (strict threshold)
    cutoff    = np.percentile(scores, 2)
    anomalies = trades[trades["anomaly_score"] < cutoff]

    for _, row in anomalies.iterrows():
        # Guess violation type from which feature is most extreme
        if   row["qty_z"] > 5:
            vtype  = "wash_trading"
            reason = f"Quantity z-score={row['qty_z']:.1f}x normal"
        elif row["price_dev"] > 0.02:
            vtype  = "spoofing"
            reason = f"Price deviates {row['price_dev']*100:.2f}% from bar mid"
        elif row["wallet_freq"] < 3:
            vtype  = "aml_structuring"
            reason = "Low-frequency wallet with anomalous trade"
        else:
            vtype  = "wash_trading"
            reason = f"Multi-feature anomaly (score={row['anomaly_score']:.3f})"

        results.append({
            "symbol":         symbol,
            "date":           row["date"],
            "trade_id":       row["trade_id"],
            "violation_type": vtype,
            "remarks": (
                f"Isolation Forest anomaly (score={row['anomaly_score']:.3f}, cutoff={cutoff:.3f}). "
                f"{reason}. qty_z={row['qty_z']:.1f}, "
                f"price_dev={row['price_dev']*100:.2f}%, "
                f"wallet_freq={row['wallet_freq']:.0f}."
            )
        })

    print(f"    → Isolation Forest: {len(results)} flags")
    return results


# =============================================================
# HELPER: Deduplicate by trade_id, keep first occurrence
# =============================================================
def _dedup(results):
    seen  = set()
    dedup = []
    for r in results:
        if r["trade_id"] not in seen:
            seen.add(r["trade_id"])
            dedup.append(r)
    return dedup


# =============================================================
# STEP 3 — MAIN ORCHESTRATOR
# Runs the right detectors for each pair and writes submission.csv
# =============================================================
def run_all():
    all_results = []

    print("\n" + "=" * 65)
    print("  PROBLEM 3 — CRYPTO ANOMALY DETECTION")
    print("=" * 65)

    for symbol in PAIRS:
        print(f"\n[{symbol}]")
        try:
            market, trades = load_pair_data(symbol)
        except FileNotFoundError:
            print(f"  ⚠  Files not found for {symbol} — skipping")
            continue

        # ── Route each symbol to the right detectors ──────────
        if symbol == "USDCUSDT":
            # Stablecoin: any price ≠ $1.00 is the primary signal
            all_results += detect_peg_break(trades, market, symbol)

        elif symbol == "BATUSDT":
            # Very illiquid: AML structuring is the primary pattern
            all_results += detect_aml_structuring(trades, market, symbol)
            all_results += detect_wash_trading(trades, market, symbol)

        elif symbol == "DOGEUSDT":
            # Sentiment-driven: pump-and-dump is very common here
            all_results += detect_pump_and_dump(trades, market, symbol)
            all_results += detect_wash_trading(trades, market, symbol)

        elif symbol == "LTCUSDT":
            # Moderate liquidity: wash trading + AML patterns
            all_results += detect_wash_trading(trades, market, symbol)
            all_results += detect_ramping(trades, market, symbol)
            all_results += detect_aml_structuring(trades, market, symbol)

        elif symbol == "XRPUSDT":
            # Fast & cheap: AML + wash trades
            all_results += detect_aml_structuring(trades, market, symbol)
            all_results += detect_wash_trading(trades, market, symbol)
            all_results += detect_isolation_forest(trades, market, symbol)

        elif symbol == "SOLUSDT":
            # Volatile: pump-and-dump + ML catch-all
            all_results += detect_pump_and_dump(trades, market, symbol)
            all_results += detect_wash_trading(trades, market, symbol)
            all_results += detect_isolation_forest(trades, market, symbol)

        elif symbol == "ETHUSDT":
            # High liquidity: layering echo + wash + ML
            all_results += detect_wash_trading(trades, market, symbol)
            all_results += detect_layering_echo(trades, market, symbol)
            all_results += detect_ramping(trades, market, symbol)
            all_results += detect_isolation_forest(trades, market, symbol)

        elif symbol == "BTCUSDT":
            # Highest liquidity: needs all detectors, tight thresholds
            all_results += detect_wash_trading(trades, market, symbol)
            all_results += detect_layering_echo(trades, market, symbol)
            all_results += detect_ramping(trades, market, symbol)
            all_results += detect_isolation_forest(trades, market, symbol)

    # ==========================================================
    # STEP 4 — BUILD AND SAVE submission.csv
    # ==========================================================
    elapsed = time.time() - start_time

    if not all_results:
        print("\n⚠  No suspicious trades found. Check DATA_DIR path.")
        return

    sub = pd.DataFrame(all_results)
    sub = sub.drop_duplicates(subset=["trade_id"], keep="first")
    sub = sub[["symbol", "date", "trade_id", "violation_type", "remarks"]]
    sub = sub.sort_values(["symbol", "date"]).reset_index(drop=True)

    out_path = "submission.csv"
    sub.to_csv(out_path, index=False)

    print("\n" + "=" * 65)
    print(f"  DONE in {elapsed:.1f}s — {len(sub):,} suspicious trades flagged")
    print(f"  Saved → {out_path}")
    print("=" * 65)

    print("\nBy violation type:")
    print(sub["violation_type"].value_counts().to_string())
    print("\nBy symbol:")
    print(sub["symbol"].value_counts().to_string())

    # ── Runtime bonus reminder ─────────────────────────────────
    if elapsed < 60:
        print(f"\n✅ Runtime {elapsed:.1f}s < 60s → eligible for +2 bonus per true positive!")
    elif elapsed < 300:
        print(f"\n⚡ Runtime {elapsed:.1f}s — under 5 min, good but aim for under 1 min next run.")
    else:
        print(f"\n⚠  Runtime {elapsed:.1f}s > 5 min — optimise with vectorised pandas operations.")

    return sub


# =============================================================
# ENTRY POINT
# =============================================================
if __name__ == "__main__":
    run_all()
