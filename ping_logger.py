#!/usr/bin/env python3
"""
net_watch.py — Continuous connectivity/quality logger to CSV.

Logs:
- ICMP ping to multiple targets (gateway + ISP-near + global)
- DNS resolution probe (optional) to detect "connected, no internet" style failures
- Sequence number + UTC timestamps for ISP-grade evidence

Runtime UX:
- Optional periodic "heartbeat" print so you see it's still running
  (disable with --quiet)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import platform
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List

# Tries to parse RTT from ping output (Windows / Linux / macOS)
RTT_RE = re.compile(r"time[=<]\s*(\d+(?:\.\d+)?)\s*ms", re.IGNORECASE)

DEFAULT_GLOBAL_TARGETS = [
    "1.1.1.1",  # Cloudflare
    "8.8.8.8",  # Google
]

# A Finland/ISP-near-ish example (you used 193.166.4.1).
# Keep it optional; you can replace or add more.
DEFAULT_ISP_NEAR = [
    "193.166.4.1",
]

DEFAULT_DNS_PROBE_HOSTS = [
    "www.google.com",
    "www.cloudflare.com",
]


def iso_utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def detect_default_gateway() -> Optional[str]:
    """
    Best-effort gateway detection, no external deps.
    Returns gateway IP as string or None.
    """
    system = platform.system().lower()

    try:
        if "windows" in system:
            # Parse: "Default Gateway . . . . . . . . . : 192.168.1.1"
            p = subprocess.run(["ipconfig"], capture_output=True, text=True, timeout=5)
            txt = p.stdout or ""
            # ipconfig can show multiple adapters; grab the first IPv4-like gateway
            m = re.search(r"Default Gateway[ .:]*\s*([\d]{1,3}(?:\.[\d]{1,3}){3})", txt)
            return m.group(1) if m else None

        # Linux: "ip route" -> "default via 192.168.1.1 dev ..."
        if "linux" in system:
            p = subprocess.run(["ip", "route"], capture_output=True, text=True, timeout=5)
            txt = p.stdout or ""
            m = re.search(r"default\s+via\s+([\d]{1,3}(?:\.[\d]{1,3}){3})", txt)
            return m.group(1) if m else None

        # macOS: "route -n get default" -> "gateway: 192.168.1.1"
        if "darwin" in system or "mac" in system:
            p = subprocess.run(["route", "-n", "get", "default"], capture_output=True, text=True, timeout=5)
            txt = p.stdout or ""
            m = re.search(r"gateway:\s+([\d]{1,3}(?:\.[\d]{1,3}){3})", txt)
            return m.group(1) if m else None

    except Exception:
        return None

    return None


def build_ping_cmd(host: str, timeout_ms: int) -> List[str]:
    system = platform.system().lower()
    if "windows" in system:
        # -n 1 = one echo, -w timeout in ms
        return ["ping", "-n", "1", "-w", str(timeout_ms), host]
    else:
        # Linux/macOS:
        # Use -c 1 and rely on subprocess timeout for portability.
        # (Linux has -W seconds, macOS differs, so keep it simple.)
        return ["ping", "-c", "1", host]


def ping_once(host: str, timeout_ms: int) -> Tuple[bool, Optional[float], str]:
    """
    Returns (ok, rtt_ms, note).
    ok False if timeout or ping returns nonzero.
    """
    cmd = build_ping_cmd(host, timeout_ms)
    t0 = time.perf_counter()
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(1.0, timeout_ms / 1000.0 + 1.0),
        )
        out = (p.stdout or "") + "\n" + (p.stderr or "")
        if p.returncode == 0:
            m = RTT_RE.search(out)
            if m:
                return True, float(m.group(1)), ""
            # Success but RTT not parsed; fall back to elapsed
            return True, (time.perf_counter() - t0) * 1000.0, "rtt_unparsed_used_elapsed"
        return False, None, f"ping_failed_rc={p.returncode}"
    except subprocess.TimeoutExpired:
        return False, None, "timeout"
    except Exception as e:
        return False, None, f"error:{type(e).__name__}:{e}"


def dns_probe(hostname: str, timeout_s: float) -> Tuple[bool, Optional[float], str]:
    """
    Resolves a hostname using system DNS (same path your apps use).
    Returns (ok, ms, note).
    """
    t0 = time.perf_counter()
    try:
        old_to = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout_s)
        try:
            # getaddrinfo triggers DNS resolution
            socket.getaddrinfo(hostname, 443, proto=socket.IPPROTO_TCP)
        finally:
            socket.setdefaulttimeout(old_to)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        return True, dt_ms, ""
    except socket.gaierror:
        return False, None, "dns_gaierror"
    except socket.timeout:
        return False, None, "dns_timeout"
    except Exception as e:
        return False, None, f"dns_error:{type(e).__name__}:{e}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Network quality logger (ping + DNS) to CSV.")
    ap.add_argument("--interval", type=float, default=1.0, help="Seconds between rounds (default 1.0).")
    ap.add_argument("--duration-hours", type=float, default=28.0, help="Run time in hours (0 = infinite).")
    ap.add_argument("--timeout-ms", type=int, default=1000, help="Ping timeout per target in ms (default 1000).")
    ap.add_argument("--out", type=str, default="net_watch.csv", help="Output CSV (default net_watch.csv).")

    ap.add_argument("--no-gateway", action="store_true", help="Do not auto-add the default gateway target.")
    ap.add_argument("--gateway", type=str, default=None, help="Manually set gateway IP (overrides auto-detect).")

    ap.add_argument("--targets", nargs="*", default=[],
                    help="Extra ping targets (IPs or hostnames).")
    ap.add_argument("--add-isp-near", action="store_true",
                    help="Add ISP-near default target(s) (includes 193.166.4.1).")
    ap.add_argument("--add-global", action="store_true",
                    help="Add global default target(s) (1.1.1.1 and 8.8.8.8).")

    ap.add_argument("--dns", action="store_true", help="Enable DNS probe logging.")
    ap.add_argument("--dns-hosts", nargs="*", default=[],
                    help="Hostnames for DNS probe (default: www.google.com www.cloudflare.com).")
    ap.add_argument("--dns-timeout", type=float, default=2.0, help="DNS probe timeout seconds (default 2.0).")

    # Heartbeat output
    ap.add_argument("--quiet", action="store_true",
                    help="Do not print periodic status (heartbeat) lines while running.")
    ap.add_argument("--heartbeat-seconds", type=float, default=60.0,
                    help="Heartbeat interval in seconds (default 60). Ignored with --quiet.")

    args = ap.parse_args()

    interval = max(0.2, args.interval)
    timeout_ms = max(100, args.timeout_ms)

    # Build target list
    ping_targets: List[str] = []

    # Gateway
    gw = None
    if not args.no_gateway:
        gw = args.gateway or detect_default_gateway()
        if gw:
            ping_targets.append(gw)

    # Defaults
    if args.add_isp_near:
        ping_targets.extend(DEFAULT_ISP_NEAR)
    if args.add_global:
        ping_targets.extend(DEFAULT_GLOBAL_TARGETS)

    # User targets
    ping_targets.extend(args.targets or [])

    # De-duplicate while preserving order
    seen = set()
    ping_targets = [t for t in ping_targets if not (t in seen or seen.add(t))]

    if not ping_targets:
        print("No ping targets configured. Try --add-global and/or --add-isp-near, and/or --targets ...", file=sys.stderr)
        return 2

    # DNS probe hosts
    dns_hosts = args.dns_hosts or (DEFAULT_DNS_PROBE_HOSTS if args.dns else [])

    out_path = Path(args.out)
    new_file = not out_path.exists()

    end_time = None
    if args.duration_hours and args.duration_hours > 0:
        end_time = time.time() + args.duration_hours * 3600.0

    if not args.quiet:
        print(f"Logging to: {out_path.resolve()}")
        print(f"Interval: {interval}s | Ping timeout: {timeout_ms} ms")
        if gw:
            print(f"Gateway target: {gw}")
        print(f"Ping targets: {', '.join(ping_targets)}")
        if args.dns:
            print(f"DNS probe: ON (timeout {args.dns_timeout}s) hosts: {', '.join(dns_hosts)}")
        else:
            print("DNS probe: OFF")
        print("Stop with Ctrl+C.")

    # CSV columns designed for later reporting/plotting
    header = ["seq", "ts_utc", "kind", "target", "ok", "value_ms", "note"]
    # kind: "ping" or "dns"
    # value_ms: RTT for ping, resolve time for dns

    seq = 0
    last_heartbeat = time.time()
    heartbeat_every = max(1.0, float(args.heartbeat_seconds))

    with out_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)

        try:
            while True:
                loop_start = time.perf_counter()
                ts = iso_utc_now()

                # Pings
                for host in ping_targets:
                    ok, rtt, note = ping_once(host, timeout_ms)
                    w.writerow([seq, ts, "ping", host, int(ok), f"{rtt:.3f}" if rtt is not None else "", note])
                    seq += 1

                # DNS probes (optional)
                if args.dns:
                    for hn in dns_hosts:
                        ok, ms, note = dns_probe(hn, args.dns_timeout)
                        w.writerow([seq, ts, "dns", hn, int(ok), f"{ms:.3f}" if ms is not None else "", note])
                        seq += 1

                f.flush()

                # --- periodic heartbeat (optional) ---
                if not args.quiet:
                    now = time.time()
                    if now - last_heartbeat >= heartbeat_every:
                        print(f"[{iso_utc_now()}] running, seq={seq}", flush=True)
                        last_heartbeat = now

                if end_time and time.time() >= end_time:
                    break

                elapsed = time.perf_counter() - loop_start
                sleep_for = interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

        except KeyboardInterrupt:
            if not args.quiet:
                print("\nStopped.")
            return 0

    if not args.quiet:
        print("Completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
