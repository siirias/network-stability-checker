#!/usr/bin/env python3
"""
make_plots.py — Generate report figures from net_watch CSV logs.

Reads CSVs from ./data/
Writes figures to ./report/

Default: use all available data
Option: --days N  (use only last N days)
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from typing import Optional


DATA_DIR = Path(__file__).parent / "data"
REPORT_DIR = Path(__file__).parent / "report"

RTT_CAP = 250.0  # ms
LOSS_BIN_S = 5   # seconds


# -----------------------------
# Data loading
# -----------------------------

def load_data(days: Optional[int]) -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("net_watch_*.csv"))
    if not files:
        raise RuntimeError("No data files found")

    dfs = []
    for fn in files:
        df = pd.read_csv(fn, parse_dates=["ts_utc"])
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df[df["kind"] == "ping"].copy()

    if days is not None:
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
        df = df[df["ts_utc"] >= cutoff]

    return df.sort_values("ts_utc")


# -----------------------------
# Figure 1: time series
# -----------------------------

def plot_timeseries(df: pd.DataFrame):
    fig, (ax_rtt, ax_loss) = plt.subplots(
        2, 1, figsize=(14, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]}
    )

    # RTT plot
    for target, g in df.groupby("target"):
        ok = g["ok"] == 1
        rtt = g.loc[ok, "value_ms"].clip(upper=RTT_CAP)

        ax_rtt.plot(
            g.loc[ok, "ts_utc"],
            rtt,
            marker=".",
            linestyle="-",
            label=target,
            alpha=0.8
        )

    ax_rtt.set_ylim(0, RTT_CAP)
    ax_rtt.set_ylabel("RTT (ms)")
    ax_rtt.set_title("Ping RTT (capped) + packet loss")
    ax_rtt.grid(True, alpha=0.3)
    ax_rtt.legend()

    # Loss strip
    df_loss = df.copy()
    df_loss["fail"] = (df_loss["ok"] == 0).astype(int)
    df_loss["bin"] = df_loss["ts_utc"].dt.floor(f"{LOSS_BIN_S}s")

    loss_rate = df_loss.groupby("bin")["fail"].mean()

    ax_loss.imshow(
        loss_rate.values.reshape(1, -1),
        aspect="auto",
        cmap="gray_r",
        vmin=0,
        vmax=1,
        extent=[
            mdates.date2num(loss_rate.index.min()),
            mdates.date2num(loss_rate.index.max()),
            0, 1,
        ],
    )

    ax_loss.set_yticks([])
    ax_loss.set_ylabel("loss")
    ax_loss.set_xlabel("UTC time")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "timeseries_rtt_loss.png", dpi=150)
    plt.close(fig)


# -----------------------------
# Figure 2: diurnal pattern
# -----------------------------

def plot_diurnal(df: pd.DataFrame):
    df_ok = df[df["ok"] == 1].copy()
    df_ok["hour"] = df_ok["ts_utc"].dt.hour

    stats = df_ok.groupby("hour")["value_ms"].agg(
        median="median",
        p95=lambda x: np.percentile(x, 95),
    )

    df_loss = df.copy()
    df_loss["hour"] = df_loss["ts_utc"].dt.hour
    loss = df_loss.groupby("hour")["ok"].apply(lambda x: 1 - x.mean())

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(stats.index, stats["median"], label="median RTT")
    ax.plot(stats.index, stats["p95"], label="p95 RTT")
    ax.plot(loss.index, loss * 100, label="loss %", linestyle="--")

    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("ms / %")
    ax.set_title("Diurnal pattern of latency and packet loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(REPORT_DIR / "diurnal_quality.png", dpi=150)
    plt.close(fig)


# -----------------------------
# Figure 3: loss heatmap
# -----------------------------

def plot_loss_heatmap(df: pd.DataFrame):
    d = df.copy()
    d["date"] = d["ts_utc"].dt.date
    d["hour"] = d["ts_utc"].dt.hour
    d["fail"] = (d["ok"] == 0).astype(int)

    heat = (
        d.groupby(["date", "hour"])["fail"]
        .mean()
        .unstack()
    )

    # Force full 0–23 hour axis
    heat = heat.reindex(columns=list(range(24)))

    # Mask missing hours (no data)
    data = heat.values.astype(float)
    data_masked = np.ma.masked_invalid(data)

    # Custom colormap: bad → good
    cmap = plt.cm.get_cmap("viridis_r").copy()
    cmap.set_bad(color="white")  # no data

    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(
        data_masked,
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))
    ax.set_xlabel("Hour of day (UTC)")

    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_ylabel("Date")

    ax.set_title("Packet loss heatmap\n(red = bad, blue = good, white = no data)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("loss fraction")

    fig.tight_layout()
    fig.savefig(REPORT_DIR / "loss_heatmap.png", dpi=150)
    plt.close(fig)


# -----------------------------
# Figure 4: outage timeline
# -----------------------------

def plot_events(df: pd.DataFrame):
    """
    Event timeline with severity classes + legend.

    Bins data in time (LOSS_BIN_S seconds), computes loss fraction per bin.
    Consecutive bins with loss > 0 form an event.

    Severity classification (defaults):
      - Outage: 100% loss for >= 10 s
      - Degraded: loss >= 10% for >= 30 s OR duration >= 120 s
      - Minor: everything else
    """
    if df.empty:
        return

    d = df.copy()
    d = d.sort_values("ts_utc")
    d["fail"] = (d["ok"] == 0).astype(int)

    # Bin to fixed cadence, compute loss fraction per bin across all targets
    d["bin"] = d["ts_utc"].dt.floor(f"{LOSS_BIN_S}s")
    loss = d.groupby("bin")["fail"].mean().sort_index()  # 0..1

    if loss.empty:
        return

    # Turn loss series into contiguous "events" where loss > 0
    events = []
    in_event = False
    start = None
    max_loss = 0.0

    bins = loss.index.to_list()
    vals = loss.values

    for t, v in zip(bins, vals):
        if v > 0 and not in_event:
            in_event = True
            start = t
            max_loss = float(v)
        elif v > 0 and in_event:
            if v > max_loss:
                max_loss = float(v)
        elif v == 0 and in_event:
            end = t  # first clean bin marks event end
            events.append((start, end, max_loss))
            in_event = False
            start = None
            max_loss = 0.0

    # If file ends during an event
    if in_event and start is not None:
        end = bins[-1] + pd.Timedelta(seconds=LOSS_BIN_S)
        events.append((start, end, max_loss))

    # Severity thresholds (tunable)
    OUTAGE_MIN_S = 10
    DEGRADED_MIN_S = 30
    DEGRADED_LOSS = 0.10
    LONG_EVENT_S = 120

    def classify(duration_s: float, max_loss_frac: float) -> str:
        if max_loss_frac >= 0.999 and duration_s >= OUTAGE_MIN_S:
            return "Outage"
        if (max_loss_frac >= DEGRADED_LOSS and duration_s >= DEGRADED_MIN_S) or (duration_s >= LONG_EVENT_S):
            return "Degraded"
        return "Minor loss"

    colors = {
        "Minor loss": "#f0c419",  # yellow-ish
        "Degraded":   "#f39c12",  # orange
        "Outage":     "#e74c3c",  # red
    }

    # Plot as horizontal bars on a time axis
    fig, ax = plt.subplots(figsize=(14, 2.4))

    for start, end, max_loss_frac in events:
        dur_s = (end - start).total_seconds()
        label = classify(dur_s, max_loss_frac)
        ax.barh(
            y=0,
            width=end - start,
            left=start,
            height=0.6,
            color=colors[label],
            edgecolor="none",
            alpha=0.9,
        )

    ax.set_yticks([])
    ax.set_xlabel("UTC time")
    ax.set_title("Loss events by severity (binned)")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()

    # Legend (explicit meaning)
    legend_handles = [
        Patch(facecolor=colors["Minor loss"], label="Minor loss (loss>0)"),
        Patch(facecolor=colors["Degraded"], label="Degraded (≥10% loss for ≥30 s or ≥120 s long)"),
        Patch(facecolor=colors["Outage"], label="Outage (100% loss for ≥10 s)"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=True)

    fig.tight_layout()
    fig.savefig(REPORT_DIR / "event_timeline.png", dpi=150)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None,
                    help="Use only last N days of data")
    args = ap.parse_args()

    REPORT_DIR.mkdir(exist_ok=True)

    df = load_data(args.days)

    plot_timeseries(df)
    plot_diurnal(df)
    plot_loss_heatmap(df)
    plot_events(df)

    print(f"Plots written to {REPORT_DIR.resolve()}")


if __name__ == "__main__":
    main()

