#!/usr/bin/env python3
"""
live_plot.py — Live ping visualizer with RTT + loss strip.

- Reads CSVs from ./data/
- Uses newest files automatically
- Plots last N minutes
- RTT plot (capped)
- Packet-loss intensity strip underneath
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


DATA_DIR = Path(__file__).parent / "data"


def load_recent_data(minutes: int) -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("net_watch_*.csv"))
    if not files:
        return pd.DataFrame()

    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(minutes=minutes)
    dfs = []

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

    return pd.concat(dfs, ignore_index=True).sort_values("ts_utc")


def plot_loop(minutes: int, refresh: float, rtt_cap: float, loss_bin_s: int):
    plt.ion()
    fig, (ax_rtt, ax_loss) = plt.subplots(
        2, 1, figsize=(12, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]}
    )

    while True:
        ax_rtt.clear()
        ax_loss.clear()

        df = load_recent_data(minutes)

        if df.empty:
            ax_rtt.set_title("No data yet")
            plt.pause(0.01)
            time.sleep(refresh)
            continue

        # --- RTT plot ---
        for target, g in df.groupby("target"):
            ok = g["ok"] == 1
            rtt = g.loc[ok, "value_ms"].clip(upper=rtt_cap)

            ax_rtt.plot(
                g.loc[ok, "ts_utc"],
                rtt,
                marker=".",
                linestyle="-",
                label=target,
            )

        ax_rtt.set_ylim(0, rtt_cap)
        ax_rtt.set_ylabel("RTT (ms)")
        ax_rtt.set_title(f"Ping RTT (capped at {rtt_cap:.0f} ms) — last {minutes} min")
        ax_rtt.grid(True, alpha=0.3)
        ax_rtt.legend(loc="upper right")

        # --- Packet loss strip (fixed) ---
        df_loss = df.copy()
        df_loss["fail"] = (df_loss["ok"] == 0).astype(int)

        df_loss["bin"] = df_loss["ts_utc"].dt.floor(f"{loss_bin_s}s")
        loss_rate = (
            df_loss.groupby("bin")["fail"]
            .mean()
            .sort_index()
        )

        for t, frac in loss_rate.items():
            if frac > 0:
                ax_loss.axvspan(
                    t,
                    t + pd.Timedelta(seconds=loss_bin_s),
                    color="black",
                    alpha=float(frac),
                    linewidth=0,
                )

        ax_loss.set_ylim(0, 1)
        ax_loss.set_yticks([])
        ax_loss.set_ylabel("loss")
        ax_loss.set_xlabel("UTC time")

        # Force identical x-limits as RTT panel
        ax_loss.set_xlim(ax_rtt.get_xlim())

        ax_loss.set_title(f"Packet loss intensity (bin = {loss_bin_s}s)")
        ax_loss.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))


        fig.autofmt_xdate()
        plt.pause(0.01)
        time.sleep(refresh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=60, help="Time window (minutes)")
    ap.add_argument("--refresh", type=float, default=5.0, help="Refresh interval (seconds)")
    ap.add_argument("--rtt-cap", type=float, default=250.0, help="Max RTT shown (ms)")
    ap.add_argument("--loss-bin", type=int, default=5, help="Loss bin size (seconds)")
    args = ap.parse_args()

    plot_loop(args.minutes, args.refresh, args.rtt_cap, args.loss_bin)


if __name__ == "__main__":
    main()
