#!/usr/bin/env python3
"""
net_stats.py — Reliability KPIs from net_watch CSV logs (Python 3.9)

Reads:  ./data/net_watch_*.csv
Prints: overall summary + hourly breakdown

Key ideas:
- Compute packet-loss per fixed time bin across ALL ping targets
- Classify each bin into: good / mediocre / poor / bad (mutually exclusive)
- Derive outage metrics from bins with 100% loss
- Compare loss for private IP targets vs public internet targets
"""

from pathlib import Path
import argparse
import ipaddress
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent / "data"

# --- defaults ---
DEFAULT_LOSS_BIN_S = 5

# Quality thresholds (loss fraction, per time bin)
# You can tune these later; they’re printed in the report.
TH_GOOD = 0.05       # < 5%
TH_MEDIOCRE = 0.15   # 5–15%
TH_POOR = 0.30       # 15–30%
# >= 30% => bad

# Peak hours (UTC)
DEFAULT_PEAK_START = 16
DEFAULT_PEAK_END = 22

# Outage definition
OUTAGE_LOSS = 0.999
DEFAULT_OUTAGE_MIN_S = 10


def is_private_ip(s: str) -> bool:
    """Return True if target string looks like a private IP (LAN/gateway)."""
    try:
        ip = ipaddress.ip_address(s)
        return ip.is_private
    except Exception:
        return False


def classify_loss(loss: float) -> str:
    """
    Map loss fraction to quality bucket.
    Buckets are mutually exclusive and sum to 100% over bins.
    """
    if np.isnan(loss):
        return "no_data"
    if loss < TH_GOOD:
        return "good"
    if loss < TH_MEDIOCRE:
        return "mediocre"
    if loss < TH_POOR:
        return "poor"
    return "bad"


def load_data(days: int = None) -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("net_watch_*.csv"))
    if not files:
        raise RuntimeError("No data files found in ./data (net_watch_*.csv)")

    dfs = []
    for fn in files:
        df = pd.read_csv(fn, parse_dates=["ts_utc"])
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df[df["kind"] == "ping"].copy()
    df = df.sort_values("ts_utc")

    if days is not None:
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
        df = df[df["ts_utc"] >= cutoff]

    return df


def compute_bins(df: pd.DataFrame, loss_bin_s: int) -> pd.DataFrame:
    """
    Return per-time-bin metrics across ALL ping targets:
    - loss fraction (mean fail over all ping attempts in that bin)
    - median RTT (only successful pings) for context
    - sample count (number of ping attempts in bin)
    """
    d = df.copy()
    d["fail"] = (d["ok"] == 0).astype(int)
    d["bin"] = d["ts_utc"].dt.floor(f"{loss_bin_s}s")

    loss = d.groupby("bin")["fail"].mean().sort_index()
    n = d.groupby("bin")["fail"].size().sort_index()
    rtt = d[d["ok"] == 1].groupby("bin")["value_ms"].median().sort_index()

    bins = pd.DataFrame({"loss": loss, "n": n, "rtt_med_ms": rtt})
    return bins


def outage_durations_seconds(bins: pd.DataFrame, loss_bin_s: int, outage_min_s: int) -> list:
    """
    Detect outages as consecutive bins with loss ~1.0.
    Returns list of outage durations in seconds (only those >= outage_min_s).
    """
    if bins.empty:
        return []

    out = []
    in_out = False
    start = None

    for t, row in bins.iterrows():
        is_out = (row["loss"] >= OUTAGE_LOSS)
        if is_out and not in_out:
            in_out = True
            start = t
        elif (not is_out) and in_out:
            end = t
            dur = (end - start).total_seconds()
            if dur >= outage_min_s:
                out.append(dur)
            in_out = False
            start = None

    if in_out and start is not None:
        # If it ends in outage, close at last bin end
        end = bins.index[-1] + pd.Timedelta(seconds=loss_bin_s)
        dur = (end - start).total_seconds()
        if dur >= outage_min_s:
            out.append(dur)

    return out


def print_definitions(loss_bin_s: int, peak_start: int, peak_end: int, outage_min_s: int):
    print("===== NETWORK RELIABILITY SUMMARY =====\n")
    print("Definitions")
    print(f"- Time resolution: {loss_bin_s}s bins")
    print("- Packet loss fraction per bin is computed across all ping targets")
    print("- Quality buckets (by loss fraction):")
    print(f"    good     : loss < {TH_GOOD*100:.0f}%")
    print(f"    mediocre : {TH_GOOD*100:.0f}% ≤ loss < {TH_MEDIOCRE*100:.0f}%")
    print(f"    poor     : {TH_MEDIOCRE*100:.0f}% ≤ loss < {TH_POOR*100:.0f}%")
    print(f"    bad      : loss ≥ {TH_POOR*100:.0f}%   (typically unusable for interactive use)")
    print(f"- Peak hours (UTC): {peak_start:02d}–{peak_end:02d}")
    print(f"- Outage: loss ~100% for ≥ {outage_min_s}s (consecutive bins)\n")


def summarize(df: pd.DataFrame, loss_bin_s: int, peak_start: int, peak_end: int, outage_min_s: int):
    bins = compute_bins(df, loss_bin_s)
    if bins.empty:
        print("No data in selected time range.")
        return

    bins["quality"] = bins["loss"].apply(classify_loss)

    # Uptime notion: “at least some packets go through”
    uptime = (bins["loss"] < 1.0).mean() * 100.0
    perfect = (bins["loss"] == 0.0).mean() * 100.0

    # Outages (100% loss)
    outs = outage_durations_seconds(bins, loss_bin_s, outage_min_s)
    outage_count = len(outs)
    outage_total_s = float(np.sum(outs)) if outs else 0.0
    outage_mean_s = float(np.mean(outs)) if outs else 0.0
    outage_max_s = float(np.max(outs)) if outs else 0.0

    # Quality distribution (exclusive)
    qdist = bins["quality"].value_counts(normalize=True) * 100.0

    # Peak-hour subset
    peak = bins[(bins.index.hour >= peak_start) & (bins.index.hour < peak_end)]
    peak_bad = (peak["quality"] == "bad").mean() * 100.0 if len(peak) else np.nan

    # Latency stats when reachable (median RTT per bin, where at least one success exists)
    rtt = bins["rtt_med_ms"].dropna()
    rtt_median = float(rtt.median()) if len(rtt) else np.nan
    rtt_p95 = float(np.percentile(rtt, 95)) if len(rtt) else np.nan

    # Local (private) vs public loss, using per-ping attempts
    d = df.copy()
    d["is_private_target"] = d["target"].astype(str).apply(is_private_ip)

    private = d[d["is_private_target"]]
    public = d[~d["is_private_target"]]

    private_loss = (private["ok"] == 0).mean() * 100.0 if len(private) else np.nan
    public_loss = (public["ok"] == 0).mean() * 100.0 if len(public) else np.nan

    # Print
    print_definitions(loss_bin_s, peak_start, peak_end, outage_min_s)

    print(f"Time span: {bins.index.min()}  →  {bins.index.max()}")
    print(f"Bins analyzed: {len(bins)}\n")

    print(f"Uptime (loss < 100%): {uptime:.1f}%")
    print(f"Perfect bins (0% loss): {perfect:.1f}%\n")

    print("Outages (100% loss)")
    print(f"  count: {outage_count}")
    print(f"  total: {outage_total_s/60.0:.1f} min")
    print(f"  mean : {outage_mean_s/60.0:.1f} min")
    print(f"  max  : {outage_max_s/60.0:.1f} min\n")

    print("Quality distribution (exclusive)")
    for k in ["good", "mediocre", "poor", "bad"]:
        print(f"  {k:8s}: {qdist.get(k, 0.0):5.1f}%")
    print()

    print(f"Peak hours (UTC {peak_start:02d}–{peak_end:02d})")
    if np.isnan(peak_bad):
        print("  bad fraction: -- (no peak-hour data)\n")
    else:
        print(f"  bad fraction: {peak_bad:.1f}%\n")

    print("Latency (successful pings, per-bin median)")
    print(f"  median RTT: {rtt_median:.1f} ms")
    print(f"  p95 RTT:    {rtt_p95:.1f} ms\n")

    print("Loss comparison (per-ping attempts)")
    print(f"  private-target loss: {private_loss:.2f}%")
    print(f"  public-target loss : {public_loss:.2f}%")
    print("\n=======================================\n")

    return bins  # for hourly report


def hourly_report(bins: pd.DataFrame):
    # If no data
    if bins is None or bins.empty:
        return

    # Compute quality per bin once (if not present)
    if "quality" not in bins.columns:
        bins = bins.copy()
        bins["quality"] = bins["loss"].apply(classify_loss)

    print("===== HOURLY QUALITY (UTC) =====")
    print("Hour |  Good% | Mediocre% |  Poor% |   Bad% | bins")
    print("---------------------------------------------------")

    for h in range(24):
        bh = bins[bins.index.hour == h]
        if len(bh) == 0:
            print(f"{h:02d}   |    -- |      -- |    -- |    -- |   0")
            continue

        vc = bh["quality"].value_counts(normalize=True) * 100.0
        print(
            f"{h:02d}   |"
            f" {vc.get('good', 0.0):6.1f} |"
            f" {vc.get('mediocre', 0.0):9.1f} |"
            f" {vc.get('poor', 0.0):6.1f} |"
            f" {vc.get('bad', 0.0):6.1f} |"
            f" {len(bh):4d}"
        )

    print("===================================================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None, help="Use only last N days")
    ap.add_argument("--loss-bin", type=int, default=DEFAULT_LOSS_BIN_S, help="Bin size (seconds)")
    ap.add_argument("--peak-start", type=int, default=DEFAULT_PEAK_START, help="Peak start hour UTC")
    ap.add_argument("--peak-end", type=int, default=DEFAULT_PEAK_END, help="Peak end hour UTC")
    ap.add_argument("--outage-min-s", type=int, default=DEFAULT_OUTAGE_MIN_S, help="Min outage duration (seconds)")
    args = ap.parse_args()

    df = load_data(args.days)
    bins = summarize(df, args.loss_bin, args.peak_start, args.peak_end, args.outage_min_s)
    hourly_report(bins)


if __name__ == "__main__":
    main()
