#!/usr/bin/env python3
"""
live_plot.py — Live ping visualizer for net_watch / ping_logger CSVs.

- Reads CSVs from ./data/
- Uses the newest files automatically
- Plots last N minutes of ping RTT
- Refreshes periodically
"""

import argparse
import time
from pathlib import Path
from datetime import timedelta

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


DATA_DIR = Path(__file__).parent / "data"


def load_recent_data(minutes: int) -> pd.DataFrame:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    files = sorted(DATA_DIR.glob("net_watch_*.csv"))
    if not files:
        return pd.DataFrame()

    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(minutes=minutes)

    dfs = []
    # Walk newest → older until we have enough time coverage
    for fn in reversed(files):
        try:
            df = pd.read_csv(fn, parse_dates=["ts_utc"])
        except Exception:
            continue

        df = df[df["kind"] == "ping"]
        df = df[df["ts_utc"] >= cutoff]

        if not df.empty:
            dfs.append(df)

        if df["ts_utc"].min() <= cutoff:
            break

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)
    return out.sort_values("ts_utc")


def plot_loop(minutes: int, refresh: float):
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 5))

    while True:
        ax.clear()

        df = load_recent_data(minutes)

        if df.empty:
            ax.set_title("No data yet")
            time.sleep(refresh)
            continue

        for target, g in df.groupby("target"):
            ok = g["ok"] == 1
            ax.plot(
                g.loc[ok, "ts_utc"],
                g.loc[ok, "value_ms"],
                marker=".",
                linestyle="-",
                label=target,
            )

            # Mark timeouts
            ax.scatter(
                g.loc[~ok, "ts_utc"],
                [0] * (~ok).sum(),
                marker="x",
                alpha=0.6,
            )

        ax.set_ylim(bottom=0)
        ax.set_ylabel("RTT (ms)")
        ax.set_xlabel("UTC time")
        ax.set_title(f"Ping RTT — last {minutes} minutes")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        fig.autofmt_xdate()

        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.pause(0.01)
        time.sleep(refresh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=60, help="Time window (minutes)")
    ap.add_argument("--refresh", type=float, default=5.0, help="Refresh interval (seconds)")
    args = ap.parse_args()

    plot_loop(args.minutes, args.refresh)


if __name__ == "__main__":
    main()

